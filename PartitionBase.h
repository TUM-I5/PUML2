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
#include "utils/logger.h"
#include "Topology.h"
#include "PartitionGraph.h"
#include "PartitionTarget.h"
#include <vector>

namespace PUML {

enum class PartitioningResult { SUCCESS = 0, ERROR };

template <TopoType Topo>
class PartitionBase {
  public:
  PartitionBase() = default;
  virtual ~PartitionBase() = default;

#ifdef USE_MPI
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
    return partition(part.data(), graph, target, seed);
  }

  virtual auto partition(int* part,
                         const PartitionGraph<Topo>& graph,
                         const PartitionTarget& target,
                         int seed = 1) -> PartitioningResult = 0;
#endif // USE_MPI
};

using TETPartitionBase = PartitionBase<TETRAHEDRON>;

} // namespace PUML

#endif // PUML_PARTITION_BASE_H
