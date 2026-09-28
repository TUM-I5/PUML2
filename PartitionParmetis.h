// SPDX-FileCopyrightText: 2017-2023 Technical University of Munich
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

#ifndef PUML_PARTITIONPARMETIS_H
#define PUML_PARTITIONPARMETIS_H

#include "PartitionTarget.h"
#include <array>
#include <vector>
#include "utils/logger.h"
#ifdef USE_MPI
#endif // USE_MPI

#ifndef USE_PARMETIS
#warning ParMETIS is not enabled.
#endif

#include <metis.h>
#include <parmetis.h>

#include <cassert>

#include "PartitionBase.h"
#include "PartitionGraph.h"
#include "Topology.h"

namespace PUML {

enum class ParmetisPartitionMode { Default, Geometric };

template <TopoType Topo>
class PartitionParmetis : public PartitionBase<Topo> {

  public:
  PartitionParmetis(ParmetisPartitionMode mode) : mode(mode) {}
#ifdef USE_MPI
  auto partition(int* partition,
                 const PartitionGraph<Topo>& graph,
                 const PartitionTarget& target,
                 int seed = 1) -> PartitioningResult override {
    auto comm = graph.comm();
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // ParMETIS refuses a graph with a rank that holds no vertices ("Poor initial vertex
    // distribution"), or no edges, since the adjacency of that rank is empty and its pointer null
    // ("adjncy is NULL"). With weights, such a rank leaves the checks before the others do, which
    // go on to sum the weights over all ranks: they wait for it forever, or take part in
    // different reductions. Every rank knows both distributions, so they all give up here
    // together.
    for (std::size_t i = 0; i < graph.processCount(); ++i) {
      if (graph.vertexDistribution()[i + 1] == graph.vertexDistribution()[i]) {
        logWarning(rank) << "ParMETIS cannot partition a graph in which a rank holds no cells, as"
                         << "rank" << i << "does.";
        return PartitioningResult::ERROR;
      }
      if (graph.edgeDistribution()[i + 1] == graph.edgeDistribution()[i]) {
        logWarning(rank) << "ParMETIS cannot partition a graph in which the cells of a rank have no"
                         << "neighbours, as those of rank" << i << "do not.";
        return PartitioningResult::ERROR;
      }
    }

    std::vector<idx_t> vtxdist(graph.vertexDistribution().begin(),
                               graph.vertexDistribution().end());
    std::vector<idx_t> xadj(graph.adjDisp().begin(), graph.adjDisp().end());
    std::vector<idx_t> adjncy(graph.adj().begin(), graph.adj().end());
    std::vector<idx_t> vwgt(graph.vertexWeights().begin(), graph.vertexWeights().end());
    std::vector<idx_t> adjwgt(graph.edgeWeights().begin(), graph.edgeWeights().end());
    auto cellCount = graph.localVertexCount();

    auto ncon = static_cast<idx_t>(graph.vertexWeightCount());
    if (ncon == 0) {
      ncon = 1;
    }
    auto nparts = static_cast<idx_t>(target.partitionCount());
    std::vector<real_t> tpwgts(static_cast<std::size_t>(nparts) * ncon,
                               static_cast<real_t>(1.) / static_cast<real_t>(nparts));
    if (!target.partitionWeightsUniform()) {
      for (std::size_t i = 0; i < target.partitionCount(); i++) {
        for (idx_t j = 0; j < ncon; ++j) {
          tpwgts[(i * static_cast<std::size_t>(ncon)) + j] =
              static_cast<real_t>(target.partitionWeights()[i]);
        }
      }
    }

    std::array<idx_t, 3> options = {1, 0, static_cast<idx_t>(seed)};
    idx_t numflag = 0;
    idx_t wgtflag = 0;
    if (graph.vertexWeightCount() > 0) {
      wgtflag |= 2;
    }
    if (graph.hasEdgeWeights()) {
      wgtflag |= 1;
    }
    auto* vwgtArray = internal::weightArray(vwgt, graph.vertexWeightCount() > 0);
    auto* adjwgtArray = internal::weightArray(adjwgt, graph.hasEdgeWeights());
    std::vector<real_t> ubvec(static_cast<std::size_t>(ncon),
                              static_cast<real_t>(target.imbalance() + 1.0));

    idx_t edgecut = 0;
    std::vector<idx_t> part(cellCount);

    int status = METIS_OK;
    if (mode == ParmetisPartitionMode::Default) {
      status = ParMETIS_V3_PartKway(vtxdist.data(),
                                    xadj.data(),
                                    adjncy.data(),
                                    vwgtArray,
                                    adjwgtArray,
                                    &wgtflag,
                                    &numflag,
                                    &ncon,
                                    &nparts,
                                    tpwgts.data(),
                                    ubvec.data(),
                                    options.data(),
                                    &edgecut,
                                    part.data(),
                                    &comm);
    } else if (mode == ParmetisPartitionMode::Geometric) {
      idx_t ndims = 3;
      std::vector<real_t> xyz;
      graph.geometricCoordinates(xyz);
      status = ParMETIS_V3_PartGeomKway(vtxdist.data(),
                                        xadj.data(),
                                        adjncy.data(),
                                        vwgtArray,
                                        adjwgtArray,
                                        &wgtflag,
                                        &numflag,
                                        &ndims,
                                        xyz.data(),
                                        &ncon,
                                        &nparts,
                                        tpwgts.data(),
                                        ubvec.data(),
                                        options.data(),
                                        &edgecut,
                                        part.data(),
                                        &comm);
    } else {
      logError() << "Unknown partitioning mode for ParMETIS";
      return PartitioningResult::ERROR;
    }
    // ParMETIS reduces the outcome of its checks over the ranks, so they all return alike.
    if (status != METIS_OK) {
      return PartitioningResult::ERROR;
    }

    for (std::size_t i = 0; i < cellCount; i++) {
      partition[i] = static_cast<int>(part[i]);
    }

    return PartitioningResult::SUCCESS;
  }
#endif // USE_MPI

  private:
  ParmetisPartitionMode mode;
};

} // namespace PUML

#endif // PUML_PARTITIONPARMETIS_H
