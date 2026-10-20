// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hdf5.h>

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include "Error.h"
#include "Hdf5Reader.h"
#include "PUML.h"
#include "PumlTest.h"
#include "Topology.h"
#include "Types.h"

namespace {

using namespace puml::test;

/// What the call raised on this rank, or nothing if it went through.
auto messageOf(const std::function<void()>& call) -> std::string {
  try {
    call();
  } catch (const PUML::Error& error) {
    return error.what();
  }
  return {};
}

/// Builds the mesh from its cells, split evenly over the ranks, and gives what
/// this rank raised. A defect has to be raised on every rank, also on the ones
/// that do not hold it, or they would wait for the others forever.
auto generate(const CubeMesh& mesh) -> std::string {
  PUML::TETPUML puml;
  feed(puml,
       mesh,
       evenSplit(mesh.numCells, commRank(), commSize()),
       evenSplit(mesh.numVertices, commRank(), commSize()));
  return messageOf([&puml]() { puml.generateMesh(); });
}

/// The rank that a message names as the one that found the defect.
auto foundOn(int rank) -> std::string {
#ifdef USE_MPI
  return "(Found on rank " + std::to_string(rank);
#else  // USE_MPI
  static_cast<void>(rank);
  return {};
#endif // USE_MPI
}

/// The first cell of the last rank, which no other rank holds.
auto cellOfTheLastRank(const CubeMesh& mesh) -> std::size_t {
  return evenSplit(mesh.numCells, commSize() - 1, commSize()).offset;
}

/// A vertex id beyond the vertices of the mesh names a vertex that no rank
/// holds; the rank with the cell has to raise it before it asks for vertices.
TEST(Defect, RaisesAVertexBeyondTheVertices) {
  auto mesh = makeCubeMesh(2);
  const auto cell = cellOfTheLastRank(mesh);
  const auto beyond = mesh.numVertices + 3;
  mesh.connect[(4 * cell) + 1] = beyond;

  const auto message = generate(mesh);
  EXPECT_NE(message.find("cell " + std::to_string(cell) + " names the vertex " +
                         std::to_string(beyond) + ", but the mesh has " +
                         std::to_string(mesh.numVertices) + " vertices"),
            std::string::npos)
      << message;
  EXPECT_NE(message.find(foundOn(commSize() - 1)), std::string::npos) << message;
}

/// Ids that count from 1 name one vertex behind the last one, and the message
/// says what that may mean.
TEST(Defect, RaisesVertexIdsCountedFromOne) {
  auto mesh = makeCubeMesh(2);
  for (auto& vertex : mesh.connect) {
    ++vertex;
  }

  // The first cell to name that vertex is the one the message names.
  const auto namesIt = [&mesh](std::size_t cell) {
    const auto first = mesh.connect.begin() + static_cast<std::ptrdiff_t>(4 * cell);
    return std::find(first, first + 4, mesh.numVertices) != first + 4;
  };
  std::size_t cell = 0;
  while (!namesIt(cell)) {
    ++cell;
  }

  const auto message = generate(mesh);
  EXPECT_NE(message.find("cell " + std::to_string(cell) + " names the vertex " +
                         std::to_string(mesh.numVertices)),
            std::string::npos)
      << message;
  EXPECT_NE(message.find("counted from 1"), std::string::npos) << message;
}

/// An id with all its bits set is what a writer may put for a vertex it does
/// not know.
TEST(Defect, RaisesAVertexIdWithAllBitsSet) {
  auto mesh = makeCubeMesh(2);
  mesh.connect[4 * cellOfTheLastRank(mesh)] = ~PUML::GlobalId{0};

  const auto message = generate(mesh);
  EXPECT_NE(message.find("All bits of the id are set"), std::string::npos) << message;
}

/// A mesh constructed from cells whose vertices were never distributed names
/// vertices that the ranks do not hold.
TEST(Defect, RaisesAVertexThatWasNotDistributed) {
  const auto mesh = makeCubeMesh(2);
  PUML::TETPUML puml;
  feed(puml,
       mesh,
       evenSplit(mesh.numCells, commRank(), commSize()),
       evenSplit(mesh.numVertices, commRank(), commSize()));
  puml.distributeVertices({});

  const auto message = messageOf([&puml]() { puml.constructMesh(puml.connectivity()); });
  EXPECT_NE(message.find("cell 0 names the vertex " + std::to_string(mesh.connect[0]) +
                         ", which this rank does not hold"),
            std::string::npos)
      << message;
  EXPECT_NE(message.find(foundOn(0)), std::string::npos) << message;
}

/// A cell listed twice, right behind itself, so that rank 0 holds both copies:
/// its faces have a third cell there, which the other ranks cannot see.
TEST(Defect, RaisesAFaceWithThreeCellsOnEveryRank) {
  auto mesh = makeCubeMesh(2);
  mesh.connect.insert(mesh.connect.begin() + 4, mesh.connect.begin(), mesh.connect.begin() + 4);
  mesh.numCells += 1;

  const auto message = generate(mesh);
  EXPECT_NE(message.find("more than two cells"), std::string::npos) << message;
  EXPECT_NE(message.find("The cells 0 and 1 have the same vertices"), std::string::npos) << message;
  EXPECT_NE(message.find("duplicated cell"), std::string::npos) << message;
  EXPECT_NE(message.find(foundOn(0)), std::string::npos) << message;
}

/// The copy of the first cell at the end, where the last rank holds it. On more
/// than one rank, rank 0 sees the first cell and its neighbours, the last rank
/// the copy, and no rank has the three cells of a face on its own.
TEST(Defect, RaisesADuplicatedCellOnAnotherRank) {
  auto mesh = makeCubeMesh(2);
  mesh.connect.insert(mesh.connect.end(), mesh.connect.begin(), mesh.connect.begin() + 4);
  mesh.numCells += 1;

  const auto message = generate(mesh);
  EXPECT_NE(message.find("more than two cells"), std::string::npos) << message;
  EXPECT_NE(message.find("duplicated cell"), std::string::npos) << message;
  EXPECT_NE(message.find(foundOn(0)), std::string::npos) << message;
  if (commSize() > 1) {
    EXPECT_NE(message.find("the cells 0 and 1 on rank 0, and a cell on rank " +
                           std::to_string(commSize() - 1)),
              std::string::npos)
        << message;
  }
}

/// A third cell on a face of two others, which is not a copy of either: the
/// first two tetrahedra of the first hexahedron share the face of its corners
/// 0, 1 and 7, and one more cell is put on it.
TEST(Defect, RaisesCellsThatOverlap) {
  auto mesh = makeCubeMesh(2);
  const std::vector<PUML::GlobalId> extra{
      mesh.connect[0], mesh.connect[1], mesh.connect[3], mesh.connect[2] + 1};
  mesh.connect.insert(mesh.connect.begin() + 8, extra.begin(), extra.end());
  mesh.numCells += 1;

  const auto message = generate(mesh);
  EXPECT_NE(message.find("more than two cells: the cells 0, 1 and 2 have it"), std::string::npos)
      << message;
  EXPECT_NE(message.find("cells that overlap"), std::string::npos) << message;
  EXPECT_EQ(message.find("same vertices"), std::string::npos) << message;
}

/// A row of zeros, which is how a part of a file reads that was never written,
/// in the first cell of the last rank.
TEST(Defect, RaisesACellOfZerosOnly) {
  auto mesh = makeCubeMesh(2);
  const auto cell = cellOfTheLastRank(mesh);
  std::fill_n(mesh.connect.begin() + static_cast<std::ptrdiff_t>(4 * cell), 4, 0);

  const auto message = generate(mesh);
  EXPECT_NE(message.find("cell " + std::to_string(cell) +
                         " names the vertex 0 more than once; its vertices are (0, 0, 0, 0)."),
            std::string::npos)
      << message;
  EXPECT_NE(message.find("never written"), std::string::npos) << message;
  EXPECT_NE(message.find(foundOn(commSize() - 1)), std::string::npos) << message;
}

/// A cell that names one of its vertices twice, in place of another one.
TEST(Defect, RaisesACellThatNamesAVertexTwice) {
  auto mesh = makeCubeMesh(2);
  const auto cell = cellOfTheLastRank(mesh);
  auto* vertices = mesh.connect.data() + (4 * cell);
  vertices[2] = vertices[0];

  const auto message = generate(mesh);
  EXPECT_NE(message.find("cell " + std::to_string(cell) + " names the vertex " +
                         std::to_string(vertices[0]) + " more than once"),
            std::string::npos)
      << message;
  EXPECT_EQ(message.find("never written"), std::string::npos) << message;
}

/// A mixed mesh pads the vertices of its smaller cells with ones they have
/// already, which is no defect; a vertex named twice among the ones a cell has
/// is one.
TEST(Defect, RaisesACellOfAMixedMeshThatNamesAVertexTwice) {
  auto mesh = makeMixedMesh(1);
  const auto cell = mesh.numHexCells + 2;
  auto* vertices = mesh.connect.data() + (8 * cell);
  vertices[3] = vertices[1];

  PUML::MIXEDPUML puml;
  feed(puml,
       mesh,
       evenSplit(mesh.numCells, commRank(), commSize()),
       evenSplit(mesh.numVertices, commRank(), commSize()));
  const auto message = messageOf([&puml]() { puml.generateMesh(); });
  EXPECT_NE(message.find("cell " + std::to_string(cell) + " names the vertex " +
                         std::to_string(vertices[1]) + " more than once"),
            std::string::npos)
      << message;
}

/// A mesh file whose cells were not all written: the rows from the first cell
/// of the last rank on were left out, and read as the fill value. Every rank
/// has to raise the first of them.
TEST(Defect, RaisesTheCellsOfAPartOfAFileThatWasNeverWritten) {
  const auto mesh = makeCubeMesh(2);
  const hsize_t missing = cellOfTheLastRank(mesh);
  const hsize_t missingRows = std::min<hsize_t>(4, mesh.numCells - missing);
  // one file per number of ranks, since the runs of the suite share a directory
  const std::string path = "puml-test-unwritten-" + std::to_string(commSize()) + ".h5";

  if (commRank() == 0) {
    // on failure, the writes fail as well, and rank 0 still meets the others
    const hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    EXPECT_GE(file, 0);

    // The space of the cells is filled with zeros when it is allocated, as a
    // file system gives them for a part of a file that was never written.
    const PUML::GlobalId zero = 0;
    const hid_t layout = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_alloc_time(layout, H5D_ALLOC_TIME_EARLY);
    H5Pset_fill_time(layout, H5D_FILL_TIME_ALLOC);
    H5Pset_fill_value(layout, H5T_NATIVE_UINT64, &zero);

    const std::array<hsize_t, 2> cellDims{mesh.numCells, 4};
    const hid_t cellSpace = H5Screate_simple(2, cellDims.data(), nullptr);
    const hid_t cells =
        H5Dcreate(file, "connect", H5T_STD_U64LE, cellSpace, H5P_DEFAULT, layout, H5P_DEFAULT);
    for (const auto& [first, count] :
         {std::array<hsize_t, 2>{0, missing},
          std::array<hsize_t, 2>{missing + missingRows, mesh.numCells - missing - missingRows}}) {
      if (count == 0) {
        continue;
      }
      const std::array<hsize_t, 2> start{first, 0};
      const std::array<hsize_t, 2> size{count, 4};
      H5Sselect_hyperslab(cellSpace, H5S_SELECT_SET, start.data(), nullptr, size.data(), nullptr);
      const hid_t memory = H5Screate_simple(2, size.data(), nullptr);
      H5Dwrite(cells, H5T_NATIVE_UINT64, memory, cellSpace, H5P_DEFAULT, &mesh.connect[4 * first]);
      H5Sclose(memory);
    }
    H5Dclose(cells);
    H5Sclose(cellSpace);
    H5Pclose(layout);

    const std::array<hsize_t, 2> vertexDims{mesh.numVertices, 3};
    const hid_t vertexSpace = H5Screate_simple(2, vertexDims.data(), nullptr);
    const hid_t vertices = H5Dcreate(
        file, "geometry", H5T_IEEE_F64LE, vertexSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(vertices, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, mesh.geometry.data());
    H5Dclose(vertices);
    H5Sclose(vertexSpace);
    H5Fclose(file);
  }
#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif // USE_MPI

  std::string message;
  {
    PUML::TETPUML puml;
#ifdef USE_MPI
    puml.setComm(MPI_COMM_WORLD);
#endif // USE_MPI
    PUML::Hdf5Reader<PUML::TETRAHEDRON> reader(puml);
    reader.open(path + ":/connect", path + ":/geometry");
    message = messageOf([&puml]() { puml.generateMesh(); });
  }

  EXPECT_NE(message.find("cell " + std::to_string(missing) +
                         " names the vertex 0 more than once; its vertices are (0, 0, 0, 0)."),
            std::string::npos)
      << message;
  EXPECT_NE(message.find("never written"), std::string::npos) << message;
  EXPECT_NE(message.find(foundOn(commSize() - 1)), std::string::npos) << message;

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif // USE_MPI
  if (commRank() == 0) {
    std::remove(path.c_str());
  }
}

} // namespace
