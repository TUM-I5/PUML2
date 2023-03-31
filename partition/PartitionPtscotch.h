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

#ifndef PUML_PARTITIONPTSCOTCH_H
#define PUML_PARTITIONPTSCOTCH_H

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <ptscotch.h>
#include <cmath>

#include "PartitionBase.h"
#include "PartitionGraph.h"
#include "Topology.h"

namespace PUML
{

template<TopoType Topo>
class PartitionPtscotch : public PartitionBase<Topo>
{

public:
	PartitionPtscotch(int mode) : mode(mode) {

	}
#ifdef USE_MPI
	virtual void partition(int* partition, const PartitionGraph<Topo>& graph, const PartitionTarget& target, int seed = 1)
	{
		assert(graph.vertex_weights().size() <= graph.local_vertex_count());

		auto comm = graph.comm();

		std::vector<SCOTCH_Num> adj_disp(graph.adj_disp().begin(), graph.adj_disp().end());
		std::vector<SCOTCH_Num> adj(graph.adj().begin(), graph.adj().end());
		std::vector<SCOTCH_Num> vertex_weights(graph.vertex_weights().begin(), graph.vertex_weights().end());
		std::vector<SCOTCH_Num> edge_weights(graph.edge_weights().begin(), graph.edge_weights().end());
		auto cell_count = graph.local_vertex_count();

		auto nparts = target.vertex_count();

		std::vector<SCOTCH_Num> weights(nparts, 1);
		if (target.has_vertex_weights()) {
			double maxv = 2;
			double minv = -1;

			for (int i = 0; i < nparts; ++i) {
				maxv = std::max(maxv, target.vertex_weights()[i]);
				minv = std::min(minv, target.vertex_weights()[i]);
			}

			double f = maxv / minv;
			f = std::min((double)(1ULL<<20), f); // prevents overflow. Adjust for more accuracy

			double scale = f / maxv;
			for (int i = 0; i < nparts; ++i) {
				weights[i] = static_cast<SCOTCH_Num>(std::round(target.vertex_weights()[i] * scale));
			}
		}

		int edgecut;
		std::vector<SCOTCH_Num> part(cell_count);

		SCOTCH_Dgraph dgraph;
		SCOTCH_Strat strategy;
		SCOTCH_Arch arch;

		SCOTCH_Num process_count = graph.process_count();
		SCOTCH_Num part_count = nparts;
		SCOTCH_Num stratflag = mode;

		int rank;
		MPI_Comm_rank(comm, &rank);

		SCOTCH_randomProc(rank);
		SCOTCH_randomSeed(seed);
		SCOTCH_randomReset();
		
		SCOTCH_dgraphInit(&dgraph, comm);
		SCOTCH_stratInit(&strategy);
		SCOTCH_archInit(&arch);

		SCOTCH_dgraphBuild(&dgraph, 0, graph.local_vertex_count(), graph.local_vertex_count(), adj_disp.data(), nullptr,
				vertex_weights.empty() ? nullptr : vertex_weights.data(), nullptr, graph.local_edge_count(), graph.local_edge_count(), 
				adj.data(), nullptr, edge_weights.empty() ? nullptr : edge_weights.data());
		SCOTCH_stratDgraphMapBuild(&strategy, stratflag, process_count, part_count, imbalance - 1);
		SCOTCH_archCmpltw(&arch, part_count, weights.data());

		SCOTCH_dgraphMap(&dgraph, &arch, &strategy, part.data());
		
		SCOTCH_archExit(&arch);
		SCOTCH_stratExit(&strategy);
		SCOTCH_dgraphExit(&dgraph);

		for (int i = 0; i < cell_count; i++) {
			partition[i] = part[i];
		}
	}
#endif // USE_MPI

private:
	int mode;
};

}

#endif // PUML_PARTITIONPTSCOTCH_H