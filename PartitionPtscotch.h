/**
 * @file
 *  This file is part of PUML
 *
 *  For conditions of distribution and use, please see the copyright
 *  notice in the file 'COPYING' at the root directory of this package
 *  and the copyright notice at https://github.com/TUM-I5/PUMGen
 *
 * @copyright 2019-2023 Technische Universitaet Muenchen
 * @author David Schneller <david.schneller@tum.de>
 */

#ifndef PUML_PARTITIONPTSCOTCH_H
#define PUML_PARTITIONPTSCOTCH_H

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#ifndef USE_PTSCOTCH
#warning PTSCOTCH is not enabled.
#endif

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
	virtual PartitioningResult partition(int* partition, const PartitionGraph<Topo>& graph, const PartitionTarget& target, int seed = 1)
	{
		int rank;
		MPI_Comm_rank(graph.comm(), &rank);

		if (graph.vertexWeights().size() > graph.localVertexCount()) {
			logWarning(rank) << "Multiple vertex weights are currently ignored by PTSCOTCH.";
		}
		if (!graph.edgeWeights().empty()) {
			logWarning(rank) << "The existence of edge weights may make PTSCOTCH very slow.";
		}

		auto comm = graph.comm();

		std::vector<SCOTCH_Num> adjDisp(graph.adjDisp().begin(), graph.adjDisp().end());
		std::vector<SCOTCH_Num> adj(graph.adj().begin(), graph.adj().end());
		std::vector<SCOTCH_Num> vertexWeights(graph.vertexWeights().begin(), graph.vertexWeights().end());
		std::vector<SCOTCH_Num> edgeWeights(graph.edgeWeights().begin(), graph.edgeWeights().end());
		auto cellCount = graph.localVertexCount();

		auto nparts = target.vertexCount();

		std::vector<SCOTCH_Num> weights(nparts, 1);
		if (!target.vertexWeightsUniform()) {
			double scale = (double)(1ULL<<24); // if this is not enough (or too much), adjust it
			for (int i = 0; i < nparts; ++i) {
				// important: the weights should be non-negative
				weights[i] = std::max(static_cast<SCOTCH_Num>(1), static_cast<SCOTCH_Num>(std::round(target.vertexWeights()[i] * scale)));
			}
		}

		int edgecut;
		std::vector<SCOTCH_Num> part(cellCount);

		SCOTCH_Dgraph dgraph;
		SCOTCH_Strat strategy;
		SCOTCH_Arch arch;

		SCOTCH_Num processCount = graph.processCount();
		SCOTCH_Num partCount = nparts;
		SCOTCH_Num stratflag = mode;

		SCOTCH_randomProc(rank);
		SCOTCH_randomSeed(seed);
		SCOTCH_randomReset();
		
		SCOTCH_dgraphInit(&dgraph, comm);
		SCOTCH_stratInit(&strategy);
		SCOTCH_archInit(&arch);

		SCOTCH_dgraphBuild(&dgraph, 0, graph.localVertexCount(), graph.localVertexCount(), adjDisp.data(), nullptr,
				vertexWeights.empty() ? nullptr : vertexWeights.data(), nullptr, graph.localEdgeCount(), graph.localEdgeCount(), 
				adj.data(), nullptr, edgeWeights.empty() ? nullptr : edgeWeights.data());
		SCOTCH_stratDgraphMapBuild(&strategy, stratflag, processCount, partCount, target.imbalance());
		SCOTCH_archCmpltw(&arch, partCount, weights.data());

		SCOTCH_dgraphMap(&dgraph, &arch, &strategy, part.data());
		
		SCOTCH_archExit(&arch);
		SCOTCH_stratExit(&strategy);
		SCOTCH_dgraphExit(&dgraph);

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

#endif // PUML_PARTITIONPTSCOTCH_H
