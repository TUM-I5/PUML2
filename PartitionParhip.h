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
 * @author David Schneller <david.schneller@tum.de>
 */

#ifndef PUML_PARTITIONPARHIP_H
#define PUML_PARTITIONPARHIP_H

#include "PartitionTarget.h"
#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#ifndef USE_PARHIP
#warning ParHIP is not enabled.
#endif

#include "utils/logger.h"

#include "PartitionBase.h"
#include "PartitionGraph.h"

#include <algorithm>
#include <cstddef>
#include <parhip_interface.h>
#include <vector>

#include "Topology.h"

namespace PUML {

template <TopoType Topo>
class PartitionParhip : public PartitionBase<Topo> {

  public:
  PartitionParhip(int mode) : mode(mode) {}
#ifdef USE_MPI
  auto partition(int* partition,
                 const PartitionGraph<Topo>& graph,
                 const PartitionTarget& target,
                 int seed = 1) -> PartitioningResult override {
    int rank = 0;
    MPI_Comm_rank(graph.comm(), &rank);

    // ParHIP does not come back for a single part; PartitionBase skips it as well, but not for
    // direct calls of this function
    if (target.partitionCount() == 1) {
      std::fill_n(partition, graph.localVertexCount(), 0);
      return PartitioningResult::SUCCESS;
    }

    std::vector<idxtype> vtxdist(graph.vertexDistribution().begin(),
                                 graph.vertexDistribution().end());
    std::vector<idxtype> xadj(graph.adjDisp().begin(), graph.adjDisp().end());
    std::vector<idxtype> adjncy(graph.adj().begin(), graph.adj().end());
    // with several weights per vertex (stored vertex by vertex), use their sum: a single weight
    // per vertex is all this partitioner takes, and the first one alone may well be zero (as
    // for the encoded balanced weights of SeisSol, which set one entry per vertex)
    const auto weightCount = std::max(graph.vertexWeightCount(), 1UL);
    std::vector<idxtype> vwgt(graph.vertexWeights().size() / weightCount);
    for (std::size_t i = 0; i < vwgt.size(); ++i) {
      for (std::size_t j = 0; j < weightCount; ++j) {
        vwgt[i] += graph.vertexWeights()[(i * weightCount) + j];
      }
    }
    std::vector<idxtype> adjwgt(graph.edgeWeights().begin(), graph.edgeWeights().end());
    auto cellCount = graph.localVertexCount();

    if (!target.partitionWeightsUniform()) {
      logWarning() << "Node weights (target vertex weights) are currently ignored by ParHIP.";
    }
    if (graph.vertexWeights().size() > graph.localVertexCount()) {
      logWarning() << "ParHIP uses the sum of multiple vertex weights.";
    }

    int edgecut = 0;
    auto nparts = static_cast<int>(target.partitionCount());
    std::vector<idxtype> part(cellCount);
    double imbalance = target.imbalance();
    MPI_Comm comm = graph.comm();
    ParHIPPartitionKWay(vtxdist.data(),
                        xadj.data(),
                        adjncy.data(),
                        vwgt.empty() ? nullptr : vwgt.data(),
                        adjwgt.empty() ? nullptr : adjwgt.data(),
                        &nparts,
                        &imbalance,
                        true,
                        seed,
                        mode,
                        &edgecut,
                        part.data(),
                        &comm);

    for (std::size_t i = 0; i < cellCount; i++) {
      partition[i] = static_cast<int>(part[i]);
    }

    return PartitioningResult::SUCCESS;
  }
#endif // USE_MPI
  private:
  int mode;
};

} // namespace PUML

#endif // PUML_PARTITIONPARHIP_H
