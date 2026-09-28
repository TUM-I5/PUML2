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
#endif // USE_MPI
#include "PartitionGraph.h"
#include "PartitionTarget.h"
#include "Topology.h"
#include "utils/logger.h"
#include <algorithm>
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
