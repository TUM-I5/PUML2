/**
 * @file
 *  This file is part of PUML
 *
 *  For conditions of distribution and use, please see the copyright
 *  notice in the file 'COPYING' at the root directory of this package
 *  and the copyright notice at https://github.com/TUM-I5/PUMGen
 *
 * @copyright 2019 Technische Universitaet Muenchen
 * @author David Schneller <david.schneller@tum.de>
 */

#ifndef PUML_PARTITIONPARHIP_H
#define PUML_PARTITIONPARHIP_H

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#ifndef USE_PARHIP
#warning ParHIP is not enabled.
#endif

#include "utils/logger.h"

#include "PartitionBase.h"
#include "PartitionGraph.h"

#include <parhip_interface.h>

#include "Topology.h"

namespace PUML
{

template<TopoType Topo>
class PartitionParhip : public PartitionBase<Topo>
{

public:
	PartitionParhip(int mode) : mode(mode) {}
#ifdef USE_MPI
	virtual PartitioningResult partition(int* partition, const PartitionGraph<Topo>& graph, const PartitionTarget& target, int seed = 1)
	{
		int rank;
		MPI_Comm_rank(graph.comm(), &rank);

		std::vector<idxtype> vtxdist(graph.vertexDistribution().begin(), graph.vertexDistribution().end());
		std::vector<idxtype> xadj(graph.adjDisp().begin(), graph.adjDisp().end());
		std::vector<idxtype> adjncy(graph.adj().begin(), graph.adj().end());
		std::vector<idxtype> vwgt(graph.vertexWeights().begin(), graph.vertexWeights().end());
		std::vector<idxtype> adjwgt(graph.edgeWeights().begin(), graph.edgeWeights().end());
		auto cellCount = graph.localVertexCount();

		if (!target.vertexWeightsUniform()) {
			logWarning(rank) << "Node weights (target vertex weights) are currently ignored by ParHIP.";
		}
		if (graph.vertexWeights().size() > graph.localVertexCount()) {
			logWarning(rank) << "Multiple vertex weights are currently ignored by ParHIP.";
		}

		int edgecut;
		int nparts = target.vertexCount();
		std::vector<idxtype> part(cellCount);
		double imbalance = target.imbalance();
		MPI_Comm comm = graph.comm();
		ParHIPPartitionKWay(vtxdist.data(), xadj.data(), adjncy.data(), vwgt.empty() ? nullptr : vwgt.data(), adjwgt.empty() ? nullptr : adjwgt.data(), &nparts, &imbalance, true, seed, mode, &edgecut, part.data(), &comm);

		for (int i = 0; i < cellCount; i++) {
			partition[i] = part[i];
		}

		return PartitioningResult::SUCCESS;
	}
#endif // USE_MPI
private:
	int mode;
};

}

#endif // PUML_PARTITIONPARHIP_H
