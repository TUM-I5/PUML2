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
	virtual void partition(int* partition, const PartitionGraph<Topo>& graph, const PartitionTarget& target, int seed = 1)
	{
		std::vector<idxtype> vtxdist(graph.vertex_distribution().begin(), graph.vertex_distribution().end());
		std::vector<idxtype> xadj(graph.adj_disp().begin(), graph.adj_disp().end());
		std::vector<idxtype> adjncy(graph.adj().begin(), graph.adj().end());
		std::vector<idxtype> vwgt(graph.vertex_weights().begin(), graph.vertex_weights().end());
		std::vector<idxtype> adjwgt(graph.edge_weights().begin(), graph.edge_weights().end());
		auto cell_count = graph.local_vertex_count();

		if (!target.vertex_weight_uniform()) {
			logWarning() << "Node weights (target vertex weights) are currently ignored by ParHIP.";
		}
		if (graph.vertex_weights().size() > graph.local_vertex_count()) {
			logWarning() << "Multiple vertex weights are currently ignored by ParHIP.";
		}

		int edgecut;
		int nparts = target.vertex_count();
		std::vector<idxtype> part(cell_count);
		double imbalance = target.imbalance();
		MPI_Comm comm = graph.comm();
		ParHIPPartitionKWay(vtxdist.data(), xadj.data(), adjncy.data(), vwgt.empty() ? nullptr : vwgt.data(), adjwgt.empty() ? nullptr : adjwgt.data(), &nparts, &imbalance, true, seed, mode, &edgecut, part.data(), &comm);

		for (int i = 0; i < cell_count; i++) {
			partition[i] = part[i];
		}
	}
#endif // USE_MPI
private:
	int mode;
};

}

#endif // PUML_PARTITIONPARHIP_H
