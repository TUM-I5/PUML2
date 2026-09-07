// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hdf5.h>

#include "Hdf5Reader.h"
#include "PumlTest.h"

namespace {

using namespace puml::test;

void writeDataset(hid_t file,
                  const char* name,
                  hid_t type,
                  const std::vector<hsize_t>& dims,
                  const void* values) {
  const hid_t space = H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
  const hid_t set = H5Dcreate(file, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(set, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values);
  H5Dclose(set);
  H5Sclose(space);
}

/// Writes a mixed mesh the way the reader expects it: one flat connectivity in
/// which every cell begins with the kind it is of, plus the offsets into it.
/// With withOffsets off, the connectivity is written as a rectangle of the one
/// kind of cell instead, which is what an older file looks like.
auto writeMesh(const MixedMesh& mesh, const std::string& path, bool withOffsets) -> void {
  if (commRank() != 0) {
    return;
  }

  const hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  ASSERT_GE(file, 0);

  writeDataset(file, "Points", H5T_NATIVE_DOUBLE, {mesh.numVertices, 3}, mesh.geometry.data());

  if (withOffsets) {
    std::vector<unsigned long> flat;
    std::vector<unsigned long> offsets{0};
    for (std::size_t c = 0; c < mesh.numCells; ++c) {
      const auto type = static_cast<PUML::CellType>(mesh.types[c]);
      const auto count = PUML::internal::shapeOf(type).vertexCount;
      flat.push_back(mesh.types[c]);
      for (unsigned int v = 0; v < count; ++v) {
        flat.push_back(mesh.connect[(8 * c) + v]);
      }
      offsets.push_back(flat.size());
    }
    writeDataset(file, "Connectivity", H5T_NATIVE_ULONG, {flat.size()}, flat.data());
    writeDataset(file, "Offsets", H5T_NATIVE_ULONG, {offsets.size()}, offsets.data());
  } else {
    const auto count = PUML::internal::shapeOf(PUML::CellType::Tetrahedron).vertexCount;
    std::vector<unsigned long> rect;
    for (std::size_t c = 0; c < mesh.numCells; ++c) {
      for (unsigned int v = 0; v < count; ++v) {
        rect.push_back(mesh.connect[(8 * c) + v]);
      }
    }
    writeDataset(file, "Connectivity", H5T_NATIVE_ULONG, {mesh.numCells, count}, rect.data());
  }

  H5Fclose(file);
}

auto scratch(const char* name) -> std::string { return std::string("puml-test-") + name + ".h5"; }

void barrier() {
#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif // USE_MPI
}

/// A file with offsets holds cells of more than one kind.
TEST(MixedReader, ReadsAMeshOfSeveralKinds) {
  const auto mesh = makeHexPyramidMesh(2);
  const auto path = scratch("mixed");
  writeMesh(mesh, path, true);
  barrier();

  PUML::MIXEDPUML puml;
#ifdef USE_MPI
  puml.setComm(MPI_COMM_WORLD);
#endif // USE_MPI
  PUML::Hdf5Reader<PUML::MIXED> reader(puml);
  reader.openMixed(path + ":/Connectivity", path + ":/Offsets", path + ":/Points");
  puml.generateMesh();

  const auto counts = measure(puml);
  EXPECT_EQ(counts.cells, static_cast<long>(mesh.numCells));
  EXPECT_EQ(counts.vertices, static_cast<long>(mesh.numVertices));
  EXPECT_EQ(counts.faces, mesh.numFaces);
  EXPECT_EQ(counts.edges, mesh.numEdges);
  EXPECT_EQ(counts.boundaryFaces, mesh.numBoundaryFaces);
  EXPECT_EQ(counts.euler(), 1);

  long triangles = 0;
  for (const auto& face : puml.faces()) {
    triangles += static_cast<long>(face.vertexCount() == 3);
  }
  EXPECT_GT(globalSum(triangles), 0);

  barrier();
  if (commRank() == 0) {
    std::remove(path.c_str());
  }
}

/// A file without offsets holds cells of one kind, which the reader is told.
TEST(MixedReader, FallsBackToTheGivenKind) {
  const auto cube = makeCubeMesh(3);
  MixedMesh mesh;
  mesh.numVertices = cube.numVertices;
  mesh.numCells = cube.numCells;
  mesh.geometry = cube.geometry;
  for (std::size_t c = 0; c < cube.numCells; ++c) {
    for (std::size_t v = 0; v < 8; ++v) {
      mesh.connect.push_back(cube.connect[(4 * c) + std::min<std::size_t>(v, 3)]);
    }
    mesh.types.push_back(static_cast<std::uint8_t>(PUML::CellType::Tetrahedron));
  }

  const auto path = scratch("uniform");
  writeMesh(mesh, path, false);
  barrier();

  PUML::MIXEDPUML puml;
#ifdef USE_MPI
  puml.setComm(MPI_COMM_WORLD);
#endif // USE_MPI
  PUML::Hdf5Reader<PUML::MIXED> reader(puml, PUML::CellType::Tetrahedron);
  reader.openMixed(path + ":/Connectivity", path + ":/Offsets", path + ":/Points");
  puml.generateMesh();

  const auto counts = measure(puml);
  EXPECT_EQ(counts.cells, static_cast<long>(cube.numCells));
  EXPECT_EQ(counts.vertices, static_cast<long>(cube.numVertices));
  EXPECT_EQ(counts.boundaryFaces, cube.numBoundaryFaces());
  EXPECT_EQ(counts.euler(), 1);

  // The same mesh read as a tetrahedral one has to give the same topology.
  PUML::TETPUML plain;
  feed(plain,
       cube,
       evenSplit(cube.numCells, commRank(), commSize()),
       evenSplit(cube.numVertices, commRank(), commSize()));
  plain.generateMesh();
  const auto reference = measure(plain);
  EXPECT_EQ(counts.faces, reference.faces);
  EXPECT_EQ(counts.edges, reference.edges);

  barrier();
  if (commRank() == 0) {
    std::remove(path.c_str());
  }
}

} // namespace
