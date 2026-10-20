// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "Error.h"
#include "PUML.h"
#include "PumlTest.h"
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

} // namespace
