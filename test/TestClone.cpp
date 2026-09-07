// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "PumlTest.h"

namespace {

using namespace puml::test;

/// A clone builds the same mesh as the original.
TEST(Clone, BuildsTheSameMesh) {
  const auto mesh = makeCubeMesh(3);
  const auto cells = evenSplit(mesh.numCells, commRank(), commSize());
  const auto vertices = evenSplit(mesh.numVertices, commRank(), commSize());

  PUML::TETPUML puml;
  feed(puml, mesh, cells, vertices);

  auto copy = puml.clone();

  puml.generateMesh();
  copy.generateMesh();

  const auto original = measure(puml);
  const auto cloned = measure(copy);

  EXPECT_EQ(original.cells, cloned.cells);
  EXPECT_EQ(original.faces, cloned.faces);
  EXPECT_EQ(original.edges, cloned.edges);
  EXPECT_EQ(original.vertices, cloned.vertices);
  EXPECT_EQ(cloned.euler(), 1);
  EXPECT_EQ(allGids(puml.faces()), allGids(copy.faces()));
}

/// The clone owns its values; writing to one does not touch the other.
TEST(Clone, OwnsItsData) {
  const auto mesh = makeCubeMesh(2);
  const auto cells = evenSplit(mesh.numCells, commRank(), commSize());

  PUML::TETPUML puml;
  feed(puml, mesh, cells, evenSplit(mesh.numVertices, commRank(), commSize()));

  std::vector<unsigned long> identity(cells.size);
  for (std::size_t i = 0; i < cells.size; ++i) {
    identity[i] = cells.offset + i;
  }
  puml.addDataArray<unsigned long>("identity", identity.data(), PUML::CELL, {});

  auto copy = puml.clone();

  // Overwrite the array in the copy only.
  auto* copied = copy.allocateData<unsigned long>("identity", PUML::CELL, {});
  for (std::size_t i = 0; i < cells.size; ++i) {
    copied[i] = 0;
  }

  const auto* untouched = reinterpret_cast<const unsigned long*>(puml.cellData("identity"));
  for (std::size_t i = 0; i < cells.size; ++i) {
    EXPECT_EQ(untouched[i], cells.offset + i) << "cell " << i;
  }
}

/// The copied arrays carry their own MPI datatype, so the clone can be
/// repartitioned on its own.
TEST(Clone, CanBeRepartitioned) {
  const auto mesh = makeCubeMesh(3);
  const auto cells = evenSplit(mesh.numCells, commRank(), commSize());
  const int procs = commSize();

  PUML::TETPUML puml;
  feed(puml, mesh, cells, evenSplit(mesh.numVertices, commRank(), commSize()));

  std::vector<unsigned long> identity(cells.size);
  for (std::size_t i = 0; i < cells.size; ++i) {
    identity[i] = cells.offset + i;
  }
  puml.addDataArray<unsigned long>("identity", identity.data(), PUML::CELL, {});

  auto copy = puml.clone();

  std::vector<int> target(cells.size);
  for (std::size_t i = 0; i < cells.size; ++i) {
    target[i] = static_cast<int>(i % procs);
  }
  copy.partition(target.data());
  copy.generateMesh();

  const auto cloned = measure(copy);
  EXPECT_EQ(cloned.cells, static_cast<long>(mesh.numCells));
  EXPECT_EQ(cloned.vertices, static_cast<long>(mesh.numVertices));
  EXPECT_EQ(cloned.boundaryFaces, mesh.numBoundaryFaces());
  EXPECT_EQ(cloned.euler(), 1);

  const auto* moved = reinterpret_cast<const unsigned long*>(copy.cellData("identity"));
  std::vector<unsigned long> local(moved, moved + copy.numOriginalCells());
  auto all = gather(local);
  std::sort(all.begin(), all.end());
  EXPECT_TRUE(isContiguousFromZero(all));

  // The original still holds its own cells, in their original order.
  const auto* untouched = reinterpret_cast<const unsigned long*>(puml.cellData("identity"));
  for (std::size_t i = 0; i < cells.size; ++i) {
    EXPECT_EQ(untouched[i], cells.offset + i) << "cell " << i;
  }
}

/// The case the clone exists for: one read, two topologies.
TEST(Clone, CarriesASecondTopology) {
  const auto mesh = makeCubeMesh(3);
  const auto cells = evenSplit(mesh.numCells, commRank(), commSize());

  PUML::TETPUML puml;
  feed(puml, mesh, cells, evenSplit(mesh.numVertices, commRank(), commSize()));

  auto topology = puml.clone();
  topology.addDataArray<unsigned long>(
      "connectivity", mesh.connect.data() + 4 * cells.offset, PUML::CELL, {4});

  puml.generateMesh();
  topology.generateMesh();

  EXPECT_EQ(measure(puml).faces, measure(topology).faces);
}

} // namespace
