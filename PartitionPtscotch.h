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

    if (graph.vertexWeightCount() > 1) {
      logWarning() << "PTSCOTCH uses the sum of multiple vertex weights.";
    }
    if (graph.hasEdgeWeights()) {
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
      unsigned long sum = 0;
      for (std::size_t j = 0; j < weightCount; ++j) {
        sum += graph.vertexWeights()[(i * weightCount) + j];
      }
      // SCOTCH_Num is a 32-bit int in many builds of PT-Scotch (Debian's and Ubuntu's among
      // them); the weights are narrowed to it like the adjacency above
      vertexWeights[i] = static_cast<SCOTCH_Num>(sum);
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

    // PT-Scotch fills in the mapping only on the ranks that pass an array for it, in a step that
    // all ranks have to take together. On a rank without cells, an empty vector need not have any
    // storage, so that rank would skip the step and leave the others waiting for it forever; the
    // array gets a dummy entry there.
    std::vector<SCOTCH_Num> part(std::max<std::size_t>(cellCount, 1));

    SCOTCH_Dgraph dgraph;
    SCOTCH_Strat strategy;
    SCOTCH_Arch arch;

    auto processCount = static_cast<SCOTCH_Num>(graph.processCount());
    auto partCount = static_cast<SCOTCH_Num>(nparts);
    auto stratflag = static_cast<SCOTCH_Num>(mode);

    SCOTCH_randomProc(rank);
    SCOTCH_randomSeed(seed);
    SCOTCH_randomReset();

    // PT-Scotch says on each rank whether a step failed there, as building the graph does where
    // the ranks disagree on whether there are weights. Building and mapping the graph are steps
    // the ranks take together, so they agree before each whether all of them may go on, and all
    // return the same.
    const auto failedAnywhere = [&comm](bool failed) {
      int anyFailed = failed ? 1 : 0;
      MPI_Allreduce(MPI_IN_PLACE, &anyFailed, 1, MPI_INT, MPI_MAX, comm);
      return anyFailed != 0;
    };

    const bool graphInitialized = SCOTCH_dgraphInit(&dgraph, comm) == 0;
    SCOTCH_stratInit(&strategy);
    SCOTCH_archInit(&arch);

    bool failed = failedAnywhere(
        !graphInitialized ||
        SCOTCH_stratDgraphMapBuild(
            &strategy, stratflag, processCount, partCount, target.imbalance()) != 0 ||
        SCOTCH_archCmpltw(&arch, partCount, weights.data()) != 0);
    if (!failed) {
      failed = failedAnywhere(
          SCOTCH_dgraphBuild(&dgraph,
                             0,
                             static_cast<SCOTCH_Num>(graph.localVertexCount()),
                             static_cast<SCOTCH_Num>(graph.localVertexCount()),
                             adjDisp.data(),
                             nullptr,
                             internal::weightArray(vertexWeights, graph.vertexWeightCount() > 0),
                             nullptr,
                             static_cast<SCOTCH_Num>(graph.localEdgeCount()),
                             static_cast<SCOTCH_Num>(graph.localEdgeCount()),
                             adj.data(),
                             nullptr,
                             internal::weightArray(edgeWeights, graph.hasEdgeWeights())) != 0);
    }
    if (!failed) {
      failed = failedAnywhere(SCOTCH_dgraphMap(&dgraph, &arch, &strategy, part.data()) != 0);
    }

    SCOTCH_archExit(&arch);
    SCOTCH_stratExit(&strategy);
    if (graphInitialized) {
      SCOTCH_dgraphExit(&dgraph);
    }

    if (failed) {
      logWarning(rank) << "PT-Scotch could not partition the graph.";
      return PartitioningResult::ERROR;
    }

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
