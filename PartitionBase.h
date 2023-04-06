/**
 * @file
 *  This file is part of PUML
 *
 *  For conditions of distribution and use, please see the copyright
 *  notice in the file 'COPYING' at the root directory of this package
 *  and the copyright notice at https://github.com/TUM-I5/PUMGen
 *
 * @copyright 2019 Technische Universitaet Muenchen
 * @author Sebastian Rettenberger <sebastian.rettenberger@tum.de>
 * @author David Schneller <david.schneller@tum.de>
 */

#ifndef PUML_PARTITION_BASE_H
#define PUML_PARTITION_BASE_H

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI
#include "utils/logger.h"
#include "Topology.h"
#include "PUML.h"
#include "PartitionGraph.h"
#include "PartitionTarget.h"
#include <vector>

namespace PUML
{

struct PartitionParameters{
	
};

template<TopoType Topo>
class PartitionBase
{
public:
	PartitionBase() { }

#ifdef USE_MPI
	std::vector<int> partition(const PartitionGraph<Topo>& graph, const PartitionTarget& target, int seed = 1)
	{
		std::vector<int> part(graph.local_vertex_count());
		partition(part, graph, target, seed);
		return part;
	}

	void partition(std::vector<int>& part, const PartitionGraph<Topo>& graph, const PartitionTarget& target, int seed = 1)
	{
		partition(part.data(), graph, target, seed);
	}

	virtual void partition(int* part, const PartitionGraph<Topo>& graph, const PartitionTarget& target, int seed = 1) = 0;

#endif // USE_MPI
};

using TETPartitionBase = PartitionBase<TETRAHEDRON>;

}

#endif // PUML_PARTITION_BASE_H
