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

#ifndef PUML_PARTITIONPARMETIS_H
#define PUML_PARTITIONPARMETIS_H

#ifdef USE_MPI
#include <mpi.h>
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

namespace PUML
{

template<TopoType Topo>
class PartitionParmetis : public PartitionBase<Topo>
{

public:
	PartitionParmetis(int mode) :mode(mode) {

	}
#ifdef USE_MPI
	virtual PartitioningResult partition(int* partition, const PartitionGraph<Topo>& graph, const PartitionTarget& target, int seed = 1)
	{
		auto comm = graph.comm();
		std::vector<idx_t> vtxdist(graph.vertexDistribution().begin(), graph.vertexDistribution().end());
		std::vector<idx_t> xadj(graph.adjDisp().begin(), graph.adjDisp().end());
		std::vector<idx_t> adjncy(graph.adj().begin(), graph.adj().end());
		std::vector<idx_t> vwgt(graph.vertexWeights().begin(), graph.vertexWeights().end());
		std::vector<idx_t> adjwgt(graph.edgeWeights().begin(), graph.edgeWeights().end());
		auto cell_count = graph.localVertexCount();

		idx_t ncon = graph.vertexWeightCount();
		if (ncon == 0) ncon = 1;
		idx_t nparts = target.vertexCount();
		std::vector<real_t> tpwgts(nparts * ncon, static_cast<real_t>(1.) / nparts);
		if (!target.vertexWeightsUniform()) {
			for (idx_t i = 0; i < target.vertexCount(); i++) {
				for (idx_t j = 0; j < ncon; ++j) {
					tpwgts[i*ncon + j] = target.vertexWeights()[i];
				}
			}
		}

		idx_t options[3] = {1, 0, seed};
		idx_t numflag = 0;
		idx_t wgtflag = 0;
		if (!vwgt.empty()) wgtflag |= 2;
		if (!adjwgt.empty()) wgtflag |= 1;
		std::vector<real_t> ubvec(ncon, target.imbalance() + 1.0);

		idx_t edgecut;
		std::vector<idx_t> part(cell_count);

		if (mode == 0) {
			ParMETIS_V3_PartKway(vtxdist.data(), xadj.data(), adjncy.data(), vwgt.empty() ? nullptr : vwgt.data(), adjwgt.empty() ? nullptr : adjwgt.data(), &wgtflag, &numflag, &ncon, &nparts, tpwgts.data(), ubvec.data(), options, &edgecut, part.data(), &comm);
		}
		else if (mode == 1) {
			idx_t ndims = 3;
			std::vector<real_t> xyz;
			graph.geometricCoordinates(xyz);
			ParMETIS_V3_PartGeomKway(vtxdist.data(), xadj.data(), adjncy.data(), vwgt.empty() ? nullptr : vwgt.data(), adjwgt.empty() ? nullptr : adjwgt.data(), &wgtflag, &numflag, &ndims, xyz.data(), &ncon, &nparts, tpwgts.data(), ubvec.data(), options, &edgecut, part.data(), &comm);
		}
		else {
			assert(false);
		}

		for (int i = 0; i < cell_count; i++) {
			partition[i] = part[i];
		}

		return PartitioningResult::SUCCESS;
	}
#endif // USE_MPI

private:
	int mode;
};

}

#endif // PUML_PARTITIONPARMETIS_H
