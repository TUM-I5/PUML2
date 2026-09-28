// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "PumlTest.h"

#include "Partition.h"
#include "PartitionGraph.h"
#include "PartitionTarget.h"

namespace {

using namespace puml::test;

/// The partitioners the build was configured with.
auto available() -> std::vector<std::pair<std::string, PUML::PartitionerType>> {
  std::vector<std::pair<std::string, PUML::PartitionerType>> partitioners{
      {"None", PUML::PartitionerType::None}};
#ifdef USE_PARMETIS
  partitioners.emplace_back("Parmetis", PUML::PartitionerType::Parmetis);
#endif // USE_PARMETIS
#ifdef USE_PTSCOTCH
  partitioners.emplace_back("PtScotch", PUML::PartitionerType::PtScotch);
  partitioners.emplace_back("PtScotchBalance", PUML::PartitionerType::PtScotchBalance);
#endif // USE_PTSCOTCH
#ifdef USE_PARHIP
  partitioners.emplace_back("ParHIPFastMesh", PUML::PartitionerType::ParHIPFastMesh);
#endif // USE_PARHIP
  return partitioners;
}

/// Builds a cube mesh, partitions it and rebuilds it on the new distribution.
void checkPartitioner(const std::string& name, PUML::PartitionerType type) {
  const int rank = commRank();
  const int procs = commSize();
  const auto mesh = makeCubeMesh(4);
  const auto cells = evenSplit(mesh.numCells, rank, procs);

  PUML::TETPUML puml;
  feed(puml, mesh, cells, evenSplit(mesh.numVertices, rank, procs));

  std::vector<unsigned long> identity(cells.size);
  for (std::size_t i = 0; i < cells.size; ++i) {
    identity[i] = cells.offset + i;
  }
  puml.addDataArray<unsigned long>("identity", identity.data(), PUML::CELL, {});

  puml.generateMesh();

  PUML::TETPartitionGraph graph(puml);
  PUML::PartitionTarget target;
  target.setPartitionCount(procs);
  target.setImbalance(0.05);

  auto partitioner = PUML::TETPartition::getPartitioner(type);
  const auto part = partitioner->partition(graph, target);

  EXPECT_EQ(part.size(), cells.size) << name;
  for (const int target : part) {
    EXPECT_GE(target, 0) << name;
    EXPECT_LT(target, procs) << name;
  }

  puml.partition(part.data());
  puml.generateMesh();

  const auto counts = measure(puml);
  EXPECT_EQ(counts.cells, static_cast<long>(mesh.numCells)) << name;
  EXPECT_EQ(counts.vertices, static_cast<long>(mesh.numVertices)) << name;
  EXPECT_EQ(counts.boundaryFaces, mesh.numBoundaryFaces()) << name;
  EXPECT_EQ(counts.euler(), 1) << name;

  // Every cell is still there exactly once, wherever it ended up.
  const auto moved = puml.data(puml.find<unsigned long>("identity", PUML::CELL));
  auto all = gather(std::vector<unsigned long>(moved.begin(), moved.end()));
  std::sort(all.begin(), all.end());
  EXPECT_EQ(all.size(), mesh.numCells) << name;
  EXPECT_TRUE(isContiguousFromZero(all)) << name;
}

TEST(Partitioner, ProducesAUsableDistribution) {
  for (const auto& [name, type] : available()) {
    checkPartitioner(name, type);
  }
}

/// The graph the partitioners are handed has to describe the mesh: one vertex
/// per cell, and an edge for every pair of cells that share a face.
TEST(Partitioner, GraphMatchesTheMesh) {
  const auto mesh = makeCubeMesh(4);
  const auto cells = evenSplit(mesh.numCells, commRank(), commSize());

  PUML::TETPUML puml;
  feed(puml, mesh, cells, evenSplit(mesh.numVertices, commRank(), commSize()));
  puml.generateMesh();

  PUML::TETPartitionGraph graph(puml);

  EXPECT_EQ(graph.localVertexCount(), cells.size);
  EXPECT_EQ(globalSum(static_cast<long>(graph.localVertexCount())),
            static_cast<long>(mesh.numCells));

  // Interior faces are the ones with two cells, and each contributes one graph
  // edge on either side.
  const auto counts = measure(puml);
  const long interior = counts.faces - counts.boundaryFaces;
  EXPECT_EQ(globalSum(static_cast<long>(graph.localEdgeCount())), 2 * interior);
}

/// Walking the edges of the graph hands every edge over once, together with the
/// cell on its far side, which may be held by another rank.
TEST(Partitioner, LocalEdgesMatchTheGraph) {
  const auto mesh = makeCubeMesh(4);

  PUML::TETPUML puml;
  feed(puml,
       mesh,
       evenSplit(mesh.numCells, commRank(), commSize()),
       evenSplit(mesh.numVertices, commRank(), commSize()));
  puml.generateMesh();

  PUML::TETPartitionGraph graph(puml);
  const auto& cells = puml.cells();
  const auto& adj = graph.adj();
  const auto& adjDisp = graph.adjDisp();

  std::vector<int> visits(graph.localEdgeCount());
  graph.forEachLocalEdges<unsigned long>(
      [&cells](int /*fid*/, int cid) { return cells[cid].gid(); },
      [&](int /*fid*/, int cid, const unsigned long& neighbor, int eid) {
        ASSERT_GE(static_cast<unsigned long>(eid), adjDisp[cid]);
        ASSERT_LT(static_cast<unsigned long>(eid), adjDisp[cid + 1]);
        EXPECT_EQ(adj[eid], neighbor);
        ++visits[eid];
      });

  for (const int count : visits) {
    EXPECT_EQ(count, 1);
  }
}

/// Without a partitioner, every cell stays on the rank that holds it.
TEST(Partitioner, NoneKeepsEveryCellWhereItIs) {
  const auto mesh = makeCubeMesh(3);

  PUML::TETPUML puml;
  feed(puml,
       mesh,
       evenSplit(mesh.numCells, commRank(), commSize()),
       evenSplit(mesh.numVertices, commRank(), commSize()));
  puml.generateMesh();

  const PUML::TETPartitionGraph graph(puml);
  PUML::PartitionTarget target;
  target.setPartitionCount(commSize());

  // The pointer variant goes to the partitioner even for a single part.
  std::vector<int> part(graph.localVertexCount(), -1);
  const auto partitioner = PUML::TETPartition::getPartitioner(PUML::PartitionerType::None);
  EXPECT_EQ(partitioner->partition(part.data(), graph, target), PUML::PartitioningResult::SUCCESS);
  for (const int owner : part) {
    EXPECT_EQ(owner, commRank());
  }
}

} // namespace
