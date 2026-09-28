// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "PumlTest.h"

#include "DataHandle.h"
#include "Error.h"
#include "PUML.h"
#include "Partition.h"
#include "PartitionBase.h"
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

/// Every part gets some of the cells: none of the meshes here is so small that
/// a partitioner would leave a part empty. One that fails without saying so and
/// hands back the partition it started from, all zeros, does.
void expectEveryPartUsed(const std::vector<int>& part, int procs, const std::string& name) {
  std::vector<long> cellsPerPart(procs);
  for (const int owner : part) {
    if (owner >= 0 && owner < procs) {
      ++cellsPerPart[owner];
    }
  }
  for (int p = 0; p < procs; ++p) {
    EXPECT_GT(globalSum(cellsPerPart[p]), 0) << name << ": part " << p << " is empty";
  }
}

/// Builds a cube mesh, partitions it and rebuilds it on the new distribution.
void checkPartitioner(const std::string& name, PUML::PartitionerType type, bool weighted) {
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
  if (weighted) {
    // Two weights for every cell, as SeisSol gives them, and one for every edge.
    graph.setVertexWeights(std::vector<int>(graph.localVertexCount() * 2, 1), 2);
    graph.setEdgeWeights(std::vector<int>(graph.localEdgeCount(), 1));
  }
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
  // Without a partitioner, every rank keeps the cells it was given, which are
  // some for each of them as well.
  expectEveryPartUsed(part, procs, name);

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
    checkPartitioner(name, type, false);
  }
}

TEST(Partitioner, ProducesAUsableDistributionWithWeights) {
  for (const auto& [name, type] : available()) {
    checkPartitioner(name, type, true);
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

  const PUML::TETPartitionGraph graph(puml);

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

/// A cell handler handed over as a temporary answers for the cells on both
/// sides of every edge, so it has to stay whole for both.
TEST(Partitioner, LocalEdgesKeepATemporaryCellHandler) {
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

  std::vector<unsigned long> gids(cells.size());
  for (std::size_t i = 0; i < cells.size(); ++i) {
    gids[i] = cells[i].gid();
  }

  std::vector<int> visits(graph.localEdgeCount());
  graph.forEachLocalEdges<unsigned long>(
      // The handler owns its values. A copy that was moved from has none, and
      // at() fails the test on it rather than reading out of bounds.
      [gids](int /*fid*/, int cid) { return gids.at(cid); },
      [&](int /*fid*/, int cid, const unsigned long& neighbor, const unsigned long& own, int eid) {
        EXPECT_EQ(own, cells[cid].gid());
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

/// What a mesh adds up to over all ranks, wherever its cells are.
struct Whole {
  long cells{0};
  long vertices{0};
  long boundaryFaces{0};
  /// The Euler characteristic, one for each piece of the mesh.
  long euler{0};
};

auto wholeOf(const CubeMesh& mesh) -> Whole {
  return {static_cast<long>(mesh.numCells),
          static_cast<long>(mesh.numVertices),
          mesh.numBoundaryFaces(),
          1};
}

/// Partitions the graph of a mesh with a rank that holds no cells, or none that
/// have a neighbour, hands the cells to the ranks they are meant for, and
/// checks that the mesh is still whole.
void repartition(PUML::TETPUML& puml,
                 const PUML::TETPartitionGraph& graph,
                 const Whole& whole,
                 const std::string& name,
                 PUML::PartitionerType type) {
  const int procs = commSize();

  PUML::PartitionTarget target;
  target.setPartitionCount(procs);
  std::vector<int> part(graph.localVertexCount(), -1);
  const auto result = PUML::TETPartition::getPartitioner(type)->partition(part, graph, target);

  // ParMETIS refuses a graph in which a rank holds no vertices ("Poor initial
  // vertex distribution") or no edges ("adjncy is NULL"), and has to say so on
  // all ranks alike.
  if (type == PUML::PartitionerType::Parmetis && procs > 1) {
    EXPECT_EQ(result, PUML::PartitioningResult::ERROR) << name;
    return;
  }

  EXPECT_EQ(result, PUML::PartitioningResult::SUCCESS) << name;
  // A partitioner that failed hands out no cells, so there is nothing to move;
  // all ranks stop if one of them has to, as moving the cells takes them all.
  if (globalMax(result == PUML::PartitioningResult::SUCCESS ? 0 : 1) != 0) {
    return;
  }
  for (const int owner : part) {
    EXPECT_GE(owner, 0) << name;
    EXPECT_LT(owner, procs) << name;
  }
  // Without a partitioner, the cells stay where they are, and not every rank
  // starts out with some.
  if (type != PUML::PartitionerType::None) {
    expectEveryPartUsed(part, procs, name);
  }

  puml.partition(part.data());
  puml.generateMesh();

  const auto counts = measure(puml);
  EXPECT_EQ(counts.cells, whole.cells) << name;
  EXPECT_EQ(counts.vertices, whole.vertices) << name;
  EXPECT_EQ(counts.boundaryFaces, whole.boundaryFaces) << name;
  EXPECT_EQ(counts.euler(), whole.euler) << name;
}

/// A rank that holds no cells has an empty part of the graph, and takes part in
/// partitioning all the same.
TEST(Partitioner, RanksWithoutCellsTakePart) {
  const int rank = commRank();
  const int procs = commSize();
  const auto mesh = makeCubeMesh(3);

  // Only the last rank starts out with cells.
  const Split cells = rank == procs - 1 ? Split{0, mesh.numCells} : Split{};

  for (const auto& [name, type] : available()) {
    PUML::TETPUML puml;
    feed(puml, mesh, cells, evenSplit(mesh.numVertices, rank, procs));
    puml.generateMesh();

    const PUML::TETPartitionGraph graph(puml);
    EXPECT_EQ(graph.localVertexCount(), cells.size) << name;
    EXPECT_EQ(graph.adjDisp().size(), cells.size + 1) << name;
    EXPECT_EQ(graph.globalVertexCount(), mesh.numCells) << name;

    repartition(puml, graph, wholeOf(mesh), name, type);
  }
}

/// The weights are a setting of the whole graph. A rank without cells has none
/// to give, but has to end up with the same setting as the others all the same,
/// or the partitioners are told different things on different ranks.
TEST(Partitioner, RanksWithoutCellsKeepTheWeights) {
  const int rank = commRank();
  const int procs = commSize();
  const auto mesh = makeCubeMesh(3);
  const Split cells = rank == procs - 1 ? Split{0, mesh.numCells} : Split{};
  constexpr int WeightCount = 2;

  for (const auto& [name, type] : available()) {
    // Once as SeisSol passes them, from the data of vectors that are empty on
    // the ranks without cells, and once as the vectors themselves.
    for (const bool asPointers : {true, false}) {
      PUML::TETPUML puml;
      feed(puml, mesh, cells, evenSplit(mesh.numVertices, rank, procs));
      puml.generateMesh();

      PUML::TETPartitionGraph graph(puml);
      const std::vector<int> vertexWeights(graph.localVertexCount() * WeightCount, 1);
      const std::vector<int> edgeWeights(graph.localEdgeCount(), 1);
      if (asPointers) {
        graph.setVertexWeights(vertexWeights.data(), WeightCount);
        graph.setEdgeWeights(edgeWeights.data());
      } else {
        graph.setVertexWeights(vertexWeights, WeightCount);
        graph.setEdgeWeights(edgeWeights);
      }

      const auto weightCount = static_cast<long>(graph.vertexWeightCount());
      EXPECT_EQ(globalMin(weightCount), WeightCount) << name;
      EXPECT_EQ(globalMax(weightCount), WeightCount) << name;
      const long edgeWeighted = graph.hasEdgeWeights() ? 1 : 0;
      EXPECT_EQ(globalMin(edgeWeighted), 1) << name;
      EXPECT_EQ(globalMax(edgeWeighted), 1) << name;

      repartition(puml, graph, wholeOf(mesh), name, type);
    }
  }
}

/// A rank whose cells have no neighbours at all holds no edges of the graph, and
/// takes part in partitioning all the same.
TEST(Partitioner, RanksWithoutEdgesTakePart) {
  const int rank = commRank();
  const int procs = commSize();

  // A cube, and one more tetrahedron beside it that shares no face with any
  // other cell. It makes a piece of the mesh of its own, with its four faces
  // on the boundary.
  const auto cube = makeCubeMesh(2);
  auto mesh = cube;
  const auto firstVertex = static_cast<unsigned long>(mesh.numVertices);
  const double x = cube.n + 1.0;
  const std::array<double, 12> corners{x, 0, 0, x + 1, 0, 0, x, 1, 0, x, 0, 1};
  mesh.geometry.insert(mesh.geometry.end(), corners.begin(), corners.end());
  for (unsigned long v = 0; v < 4; ++v) {
    mesh.connect.push_back(firstVertex + v);
  }
  mesh.numVertices += 4;
  mesh.numCells += 1;
  const Whole whole{static_cast<long>(mesh.numCells),
                    static_cast<long>(mesh.numVertices),
                    cube.numBoundaryFaces() + 4,
                    2};

  // The last rank holds the lone tetrahedron and nothing else, and the others
  // share the cube.
  Split cells{0, mesh.numCells};
  if (procs > 1) {
    cells = rank == procs - 1 ? Split{cube.numCells, 1} : evenSplit(cube.numCells, rank, procs - 1);
  }

  for (const auto& [name, type] : available()) {
    for (const bool weighted : {false, true}) {
      PUML::TETPUML puml;
      feed(puml, mesh, cells, evenSplit(mesh.numVertices, rank, procs));
      puml.generateMesh();

      PUML::TETPartitionGraph graph(puml);
      if (procs > 1 && rank == procs - 1) {
        EXPECT_EQ(graph.localVertexCount(), 1UL) << name;
        EXPECT_EQ(graph.localEdgeCount(), 0UL) << name;
      }
      if (weighted) {
        graph.setVertexWeights(std::vector<int>(graph.localVertexCount() * 2, 1), 2);
        graph.setEdgeWeights(std::vector<int>(graph.localEdgeCount(), 1));
      }

      repartition(puml, graph, whole, name + (weighted ? " with weights" : ""), type);
    }
  }
}

#ifdef USE_PTSCOTCH
/// PT-Scotch refuses a graph that has weights on some of the ranks only. Every
/// rank has to hear of it, rather than go on with the partition it started from.
TEST(Partitioner, PtScotchReportsARefusedGraph) {
  const int rank = commRank();
  const int procs = commSize();
  if (procs == 1) {
    GTEST_SKIP() << "a single rank cannot disagree with the others";
  }

  const auto mesh = makeCubeMesh(2);
  PUML::TETPUML puml;
  feed(puml, mesh, evenSplit(mesh.numCells, rank, procs), evenSplit(mesh.numVertices, rank, procs));
  puml.generateMesh();

  PUML::PartitionTarget target;
  target.setPartitionCount(procs);

  for (const bool vertexWeights : {true, false}) {
    // Against the rules, only the last rank sets the weights.
    PUML::TETPartitionGraph graph(puml);
    if (rank == procs - 1) {
      if (vertexWeights) {
        graph.setVertexWeights(std::vector<int>(graph.localVertexCount(), 1), 1);
      } else {
        graph.setEdgeWeights(std::vector<int>(graph.localEdgeCount(), 1));
      }
    }

    for (const auto type :
         {PUML::PartitionerType::PtScotch, PUML::PartitionerType::PtScotchBalance}) {
      std::vector<int> part(graph.localVertexCount(), -1);
      EXPECT_EQ(PUML::TETPartition::getPartitioner(type)->partition(part, graph, target),
                PUML::PartitioningResult::ERROR)
          << (vertexWeights ? "vertex weights" : "edge weights");
    }
  }
}
#endif // USE_PTSCOTCH

/// Weights that fall short of the local part of the graph are refused, rather
/// than read past their end or, if there are none, taken for no weights.
TEST(Partitioner, WeightsHaveToCoverTheGraph) {
  const auto mesh = makeCubeMesh(2);

  PUML::TETPUML puml;
  feed(puml,
       mesh,
       evenSplit(mesh.numCells, commRank(), commSize()),
       evenSplit(mesh.numVertices, commRank(), commSize()));
  puml.generateMesh();

  // Every rank holds cells, and each of them has neighbours.
  PUML::TETPartitionGraph graph(puml);
  const std::vector<int> oneEach(graph.localVertexCount(), 1);
  const std::vector<int> tooFewEdges(graph.localEdgeCount() - 1, 1);
  const int* none = nullptr;

  EXPECT_THROW(graph.setVertexWeights(oneEach, 2), PUML::Error);
  EXPECT_THROW(graph.setVertexWeights(oneEach, -1), PUML::Error);
  EXPECT_THROW(graph.setVertexWeights(none, 1), PUML::Error);
  EXPECT_THROW(graph.setEdgeWeights(tooFewEdges), PUML::Error);
  EXPECT_THROW(graph.setEdgeWeights(none), PUML::Error);
  EXPECT_EQ(graph.vertexWeightCount(), 0UL);
  EXPECT_FALSE(graph.hasEdgeWeights());

  graph.setVertexWeights(oneEach, 1);
  EXPECT_EQ(graph.vertexWeightCount(), 1UL);
  graph.setVertexWeights(none, 0);
  EXPECT_EQ(graph.vertexWeightCount(), 0UL);
  EXPECT_TRUE(graph.vertexWeights().empty());
}

} // namespace
