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

#ifndef PUML_PARTITIONPTSCOTCH_H
#define PUML_PARTITIONPTSCOTCH_H

#include "PartitionTarget.h"
#include "utils/logger.h"
#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#ifndef USE_PTSCOTCH
#warning PTSCOTCH is not enabled.
#endif

#include <stdint.h>
#include <stddef.h>
#include <ptscotch.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "PartitionBase.h"
#include "PartitionGraph.h"
#include "Topology.h"

namespace PUML {

template <TopoType Topo>
class PartitionPtscotch : public PartitionBase<Topo> {

  public:
  PartitionPtscotch(int mode) : mode(mode) {}
#ifdef USE_MPI
  auto partition(int* partition,
                 const PartitionGraph<Topo>& graph,
                 const PartitionTarget& target,
                 int seed = 1) -> PartitioningResult override {
    int rank = 0;
    MPI_Comm_rank(graph.comm(), &rank);

    if (graph.vertexWeights().size() > graph.localVertexCount()) {
      logWarning() << "PTSCOTCH uses the sum of multiple vertex weights.";
    }
    if (!graph.edgeWeights().empty()) {
      logWarning() << "The existence of edge weights may make PTSCOTCH very slow.";
    }

    auto comm = graph.comm();

    std::vector<SCOTCH_Num> adjDisp(graph.adjDisp().begin(), graph.adjDisp().end());
    std::vector<SCOTCH_Num> adj(graph.adj().begin(), graph.adj().end());
    // with several weights per vertex (stored vertex by vertex), use their sum: a single weight
    // per vertex is all this partitioner takes, and the first one alone may well be zero (as
    // for the encoded balanced weights of SeisSol, which set one entry per vertex)
    const auto weightCount = std::max(graph.vertexWeightCount(), 1UL);
    std::vector<SCOTCH_Num> vertexWeights(graph.vertexWeights().size() / weightCount);
    for (std::size_t i = 0; i < vertexWeights.size(); ++i) {
      for (std::size_t j = 0; j < weightCount; ++j) {
        vertexWeights[i] += graph.vertexWeights()[(i * weightCount) + j];
      }
    }
    std::vector<SCOTCH_Num> edgeWeights(graph.edgeWeights().begin(), graph.edgeWeights().end());
    auto cellCount = graph.localVertexCount();

    auto nparts = target.partitionCount();

    std::vector<SCOTCH_Num> weights(nparts, 1);
    if (!target.partitionWeightsUniform()) {
      // we need to convert from double node weights to integer node weights
      // (that is due to the interface still being oriented at ParMETIS right now)

      auto scale = (double)(1ULL << 24); // if this is not enough (or too much), adjust it
      for (std::size_t i = 0; i < nparts; ++i) {
        // important: the weights should be non-negative
        weights[i] =
            std::max(static_cast<SCOTCH_Num>(1),
                     static_cast<SCOTCH_Num>(std::round(target.partitionWeights()[i] * scale)));
      }
    }

    std::vector<SCOTCH_Num> part(cellCount);

    SCOTCH_Dgraph dgraph;
    SCOTCH_Strat strategy;
    SCOTCH_Arch arch;

    auto processCount = static_cast<SCOTCH_Num>(graph.processCount());
    auto partCount = static_cast<SCOTCH_Num>(nparts);
    auto stratflag = static_cast<SCOTCH_Num>(mode);

    SCOTCH_randomProc(rank);
    SCOTCH_randomSeed(seed);
    SCOTCH_randomReset();

    SCOTCH_dgraphInit(&dgraph, comm);
    SCOTCH_stratInit(&strategy);
    SCOTCH_archInit(&arch);

    SCOTCH_dgraphBuild(&dgraph,
                       0,
                       static_cast<SCOTCH_Num>(graph.localVertexCount()),
                       static_cast<SCOTCH_Num>(graph.localVertexCount()),
                       adjDisp.data(),
                       nullptr,
                       vertexWeights.empty() ? nullptr : vertexWeights.data(),
                       nullptr,
                       static_cast<SCOTCH_Num>(graph.localEdgeCount()),
                       static_cast<SCOTCH_Num>(graph.localEdgeCount()),
                       adj.data(),
                       nullptr,
                       edgeWeights.empty() ? nullptr : edgeWeights.data());
    SCOTCH_stratDgraphMapBuild(&strategy, stratflag, processCount, partCount, target.imbalance());
    SCOTCH_archCmpltw(&arch, partCount, weights.data());

    SCOTCH_dgraphMap(&dgraph, &arch, &strategy, part.data());

    SCOTCH_archExit(&arch);
    SCOTCH_stratExit(&strategy);
    SCOTCH_dgraphExit(&dgraph);

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

#endif // PUML_PARTITIONPTSCOTCH_H
