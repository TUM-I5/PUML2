// SPDX-FileCopyrightText: 2019-2023 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @file
 *  This file is part of PUML
 *
 *  For conditions of distribution and use, please see the copyright
 *  notice in the file 'COPYING' at the root directory of this package
 *  and the copyright notice at https://github.com/TUM-I5/PUMGen
 *
 * @author Sebastian Rettenberger <sebastian.rettenberger@tum.de>
 * @author David Schneller <david.schneller@tum.de>
 */

#ifndef PUML_PARTITION_BASE_H
#define PUML_PARTITION_BASE_H

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI
#include "PartitionGraph.h"
#include "PartitionTarget.h"
#include "Topology.h"
#include "TypeInference.h"
#include "Types.h"
#include "utils/logger.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace PUML {

enum class PartitioningResult { SUCCESS = 0, ERROR };

namespace internal {

/**
 * The weights to hand to a partitioner: nullptr if the graph has none, their array otherwise.
 *
 * The partitioners tell from the pointer whether there are weights, and the ranks have to agree
 * on it: PT-Scotch refuses a graph on which they disagree, ParHIP reduces over all ranks only on
 * the ranks that pass weights, and ParMETIS wants an array wherever its flags announce weights. A
 * rank without cells (or without edges) has no weights to pass, and an empty vector need not have
 * any storage, so the array gets a dummy entry there.
 */
template <typename T>
auto weightArray(std::vector<T>& weights, bool present) -> T* {
  if (!present) {
    return nullptr;
  }
  if (weights.empty()) {
    weights.resize(1);
  }
  return weights.data();
}

#ifdef USE_MPI
/**
 * Whether a partitioner that counts in Index can take the graph: it keeps the number of vertices
 * and of edges in it, and sums the weights in it as well -- the vertex weights each on their own,
 * or, where the partitioner adds up the weights of a vertex, all of them together. Without the
 * check, a value too large for Index wraps around when the graph is handed over, and the
 * partitioner works on another graph than the one it was given, or fails without saying why.
 *
 * Collective; all ranks come to the same answer, and the first one says what does not fit, with
 * `remedy` for how to build the partitioner so that it does.
 */
template <typename Index, TopoType Topo>
auto fitsIndexType(const PartitionGraph<Topo>& graph,
                   bool addsUpVertexWeights,
                   const char* partitioner,
                   const char* remedy) -> bool {
  constexpr auto Largest = std::numeric_limits<std::uint64_t>::max();
  constexpr auto Max = static_cast<std::uint64_t>(std::numeric_limits<Index>::max());

  // The weight sums, one per constraint the partitioner keeps, and the one of the edges behind
  // them; added up so that they stop at the largest value rather than wrap around. The ranks may
  // disagree on the number of weights, in a graph that the partitioner is to refuse itself; the
  // sums are reduced over as many as any rank has, so that all ranks reduce the same.
  Size weightCount = graph.vertexWeightCount();
  MPI_Allreduce(MPI_IN_PLACE, &weightCount, 1, MPITypeInfer<Size>::type(), MPI_MAX, graph.comm());
  const auto columns = addsUpVertexWeights ? std::min<Size>(weightCount, 1) : weightCount;
  std::vector<std::uint64_t> sums(columns + 1);
  const auto add = [](std::uint64_t& sum, std::uint64_t value) {
    sum = (sum > Largest - value) ? Largest : sum + value;
  };
  for (std::size_t i = 0; i < graph.vertexWeights().size(); ++i) {
    add(sums[addsUpVertexWeights ? 0 : i % graph.vertexWeightCount()], graph.vertexWeights()[i]);
  }
  for (const auto weight : graph.edgeWeights()) {
    add(sums.back(), weight);
  }
  // past what 64 bits hold, a sum may wrap around here after all; a graph of that weight is out of
  // reach anyway
  MPI_Allreduce(MPI_IN_PLACE,
                sums.data(),
                static_cast<int>(sums.size()),
                MPITypeInfer<std::uint64_t>::type(),
                MPI_SUM,
                graph.comm());

  int rank = 0;
  MPI_Comm_rank(graph.comm(), &rank);
  const auto fits = [&](std::uint64_t value, const char* what) {
    if (value > Max) {
      logWarning(rank) << partitioner << "counts in integers of" << 8 * sizeof(Index)
                       << "bits, which cannot hold the" << what << "of the graph:" << value
                       << "is more than" << Max << "--" << remedy;
      return false;
    }
    return true;
  };

  bool all = fits(graph.globalVertexCount(), "number of vertices");
  all = fits(graph.globalEdgeCount(), "number of edges") && all;
  for (std::size_t i = 0; i < columns; ++i) {
    all = fits(sums[i], "sum of the vertex weights") && all;
  }
  return fits(sums.back(), "sum of the edge weights") && all;
}
#endif // USE_MPI

} // namespace internal

template <TopoType Topo>
class PartitionBase {
  public:
  PartitionBase() = default;
  virtual ~PartitionBase() = default;

  auto partition(const PartitionGraph<Topo>& graph, const PartitionTarget& target, int seed = 1)
      -> std::vector<int> {
    std::vector<int> part(graph.localVertexCount());
    auto result = partition(part, graph, target, seed);
    if (result != PartitioningResult::SUCCESS) {
      logError() << "Partitioning failed.";
    }
    return part;
  }

  auto partition(std::vector<int>& part,
                 const PartitionGraph<Topo>& graph,
                 const PartitionTarget& target,
                 int seed = 1) -> PartitioningResult {
    // a single part needs no partitioner, and not all of them return for it (ParHIP does not)
    if (target.partitionCount() == 1) {
      std::fill(part.begin(), part.end(), 0);
      return PartitioningResult::SUCCESS;
    }
    return partition(part.data(), graph, target, seed);
  }

  virtual auto partition(int* part,
                         const PartitionGraph<Topo>& graph,
                         const PartitionTarget& target,
                         int seed = 1) -> PartitioningResult = 0;
};

using TETPartitionBase = PartitionBase<TETRAHEDRON>;

} // namespace PUML

#endif // PUML_PARTITION_BASE_H
