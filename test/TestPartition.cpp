// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "PumlTest.h"

namespace {

using namespace puml::test;

/// Repartitions a cube mesh and checks that the cell data travelled with its
/// cell and that the topology is unchanged.
void checkRepartition(int n, const std::vector<int>& (*assign)(std::size_t, int, int)) {
  const int rank = commRank();
  const int procs = commSize();
  const auto mesh = makeCubeMesh(n);
  const auto cells = evenSplit(mesh.numCells, rank, procs);

  PUML::TETPUML puml;
  feed(puml, mesh, cells, evenSplit(mesh.numVertices, rank, procs));

  // Two arrays whose values are derived from the same cell identity.
  std::vector<unsigned long> identity(cells.size);
  std::vector<unsigned long> derived(cells.size);
  for (std::size_t i = 0; i < cells.size; ++i) {
    identity[i] = cells.offset + i;
    derived[i] = 2 * (cells.offset + i) + 1;
  }
  puml.addDataArray<unsigned long>("identity", identity.data(), PUML::CELL, {});
  puml.addDataArray<unsigned long>("derived", derived.data(), PUML::CELL, {});

  const auto& target = assign(cells.size, rank, procs);
  puml.partition(target.data());
  puml.generateMesh();

  const auto counts = measure(puml);
  EXPECT_EQ(counts.cells, static_cast<long>(mesh.numCells));
  EXPECT_EQ(counts.vertices, static_cast<long>(mesh.numVertices));
  EXPECT_EQ(counts.boundaryFaces, mesh.numBoundaryFaces());
  EXPECT_EQ(counts.euler(), 1);

  // Every cell is still there exactly once, and its two arrays still belong
  // together.
  const auto* movedIdentity = reinterpret_cast<const unsigned long*>(puml.cellData("identity"));
  const auto* movedDerived = reinterpret_cast<const unsigned long*>(puml.cellData("derived"));

  std::vector<unsigned long> local(puml.numOriginalCells());
  for (std::size_t i = 0; i < puml.numOriginalCells(); ++i) {
    local[i] = movedIdentity[i];
    EXPECT_EQ(movedDerived[i], 2 * movedIdentity[i] + 1) << "cell " << i;
  }

  auto all = gather(local);
  std::sort(all.begin(), all.end());
  EXPECT_EQ(all.size(), mesh.numCells);
  EXPECT_TRUE(isContiguousFromZero(all));
}

/// Keeps every cell where it is.
auto keep(std::size_t size, int rank, int /*procs*/) -> const std::vector<int>& {
  static thread_local std::vector<int> target;
  target.assign(size, rank);
  return target;
}

/// Spreads the cells of every rank over all ranks, round robin.
auto scatter(std::size_t size, int /*rank*/, int procs) -> const std::vector<int>& {
  static thread_local std::vector<int> target;
  target.resize(size);
  for (std::size_t i = 0; i < size; ++i) {
    target[i] = static_cast<int>(i % procs);
  }
  return target;
}

/// Moves everything onto the last rank.
auto collect(std::size_t size, int /*rank*/, int procs) -> const std::vector<int>& {
  static thread_local std::vector<int> target;
  target.assign(size, procs - 1);
  return target;
}

TEST(Partition, IdentityKeepsTheMesh) { checkRepartition(3, keep); }

TEST(Partition, ScatteringPreservesTheMesh) { checkRepartition(3, scatter); }

TEST(Partition, CollectingOnOneRankPreservesTheMesh) { checkRepartition(3, collect); }

} // namespace
