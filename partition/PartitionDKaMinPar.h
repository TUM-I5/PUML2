/**
 * @file
 *  This file is part of PUML
 *
 *  For conditions of distribution and use, please see the copyright
 *  notice in the file 'COPYING' at the root directory of this package
 *  and the copyright notice at https://github.com/TUM-I5/PUMGen
 *
 * @copyright 2023 Technische Universitaet Muenchen
 * @author David Schneller <david.schneller@tum.de>
 */

#ifndef PUML_PARTITIONDKAMINPAR_H
#define PUML_PARTITIONDKAMINPAR_H

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI
#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "utils/logger.h"

#include "PartitionBase.h"
#include "PartitionGraph.h"

#include <dkaminpar/dkaminpar.h>

#include "Topology.h"

namespace PUML
{

template<TopoType Topo>
class PartitionDKaMinPar : public PartitionBase<Topo>
{

public:
	PartitionDKaMinPar() {}
#ifdef USE_MPI
	virtual void partition(int* partition, const PartitionGraph<Topo>& graph, const PartitionTarget& target, int seed = 1)
	{
		kaminpar::dKaMinPar dist(MPI_Comm comm, int num_threads, dist::create_default_context());

		std::vector<kaminpar::GlobalNodeID> vtxdist(graph.vertex_distribution().begin(), graph.vertex_distribution().end());
		std::vector<kaminpar::GlobalEdgeID> xadj(graph.adj_disp().begin(), graph.adj_disp().end());
		std::vector<kaminpar::GlobalNodeID> adjncy(graph.adj().begin(), graph.adj().end());
		std::vector<kaminpar::GlobalNodeWeight> vwgt(graph.vertex_weights().begin(), graph.vertex_weights().end());
		std::vector<kaminpar::GlobalEdgeWeight> adjwgt(graph.edge_weights().begin(), graph.edge_weights().end());
		auto cell_count = graph.local_vertex_count();

		if (target.has_vertex_weights()) {
			logWarning() << "Node weights (target vertex weights) are currently ignored by dKaMinPar.";
		}
		if (graph.vertex_weights().size() > graph.local_vertex_count()) {
			logWarning() << "Multiple vertex weights are currently ignored by dKaMinPar.";
		}

		kaminpar::BlockID nparts = target.vertex_count();
		std::vector<kaminpar::BlockID> part(cell_count);

#ifdef USE_OPENMP
		unsigned num_threads = omp_get_num_threads();
#else
		unsigned num_threads = 1;
#endif

		// take OMP number of threads for now
		kaminpar::dKaMinPar dist(graph.comm(), num_threads, kaminpar::dist::create_default_context());
		dist.import_graph(vtxdist.data(), xadj.data(), adjncy.data(), vwgt.empty() ? nullptr : vwgt.data(), adjwgt.empty() ? nullptr : adjwgt.data());
		dist.compute_partition(seed, nparts, part.data());

		for (int i = 0; i < cell_count; i++) {
			partition[i] = part[i];
		}
	}
#endif // USE_MPI
};

}

#endif // PUML_PARTITIONPARHIP_H
