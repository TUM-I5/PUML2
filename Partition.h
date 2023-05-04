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

#ifndef PUML_PARTITION_H
#define PUML_PARTITION_H

#include <memory>

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include "PartitionBase.h"
#include "PartitionGraph.h"
#include "PartitionDummy.h"

#ifdef USE_PARMETIS
#include "PartitionParmetis.h"
#endif

#ifdef USE_PTSCOTCH
#include "PartitionPtscotch.h"
#endif

#ifdef USE_PARHIP
#include "PartitionParhip.h"
#endif

namespace PUML
{
	
enum class PartitionerType {
	None,
	Parmetis,
	ParmetisGeometric,
	PtScotch,
	PtScotchQuality,
	PtScotchBalance,
	PtScotchBalanceQuality,
	PtScotchSpeed,
	PtScotchBalanceSpeed,
	ParHIPUltrafastMesh,
	ParHIPFastMesh,
	ParHIPEcoMesh,
	ParHIPUltrafastSocial,
	ParHIPFastSocial,
	ParHIPEcoSocial
};

template<TopoType Topo>
class Partition{
public:
static std::unique_ptr<PartitionBase<Topo>> getPartitioner(PartitionerType partitioner) {
	PartitionBase<Topo>* partition = nullptr;
	if (partitioner == PartitionerType::None) {
		partition = new PartitionDummy<Topo>();
	}
#ifdef USE_PARMETIS
	else if (partitioner == PartitionerType::Parmetis) {
		partition = new PartitionParmetis<Topo>(ParmetisPartitionMode::Default);
	}
	else if (partitioner == PartitionerType::ParmetisGeometric) {
		partition = new PartitionParmetis<Topo>(ParmetisPartitionMode::Geometric);
	}
#endif
#ifdef USE_PTSCOTCH
	else if (partitioner == PartitionerType::PtScotch) {
		partition = new PartitionPtscotch<Topo>(SCOTCH_STRATDEFAULT);
	}
	else if (partitioner == PartitionerType::PtScotchQuality) {
		partition = new PartitionPtscotch<Topo>(SCOTCH_STRATQUALITY);
	}
	else if (partitioner == PartitionerType::PtScotchBalance) {
		partition = new PartitionPtscotch<Topo>(SCOTCH_STRATBALANCE);
	}
	else if (partitioner == PartitionerType::PtScotchBalanceQuality) {
		partition = new PartitionPtscotch<Topo>(SCOTCH_STRATBALANCE | SCOTCH_STRATQUALITY);
	}
	else if (partitioner == PartitionerType::PtScotchSpeed) {
		partition = new PartitionPtscotch<Topo>(SCOTCH_STRATSPEED);
	}
	else if (partitioner == PartitionerType::PtScotchBalanceSpeed) {
		partition = new PartitionPtscotch<Topo>(SCOTCH_STRATBALANCE | SCOTCH_STRATSPEED);
	}
#endif
#ifdef USE_PARHIP
	else if (partitioner == PartitionerType::ParHIPUltrafastMesh) {
		partition = new PartitionParhip<Topo>(ULTRAFASTMESH);
	}
	else if (partitioner == PartitionerType::ParHIPFastMesh) {
		partition = new PartitionParhip<Topo>(FASTMESH);
	}
	else if (partitioner == PartitionerType::ParHIPEcoMesh) {
		partition = new PartitionParhip<Topo>(ECOMESH);
	}
	else if (partitioner == PartitionerType::ParHIPUltrafastSocial) {
		partition = new PartitionParhip<Topo>(ULTRAFASTSOCIAL);
	}
	else if (partitioner == PartitionerType::ParHIPFastSocial) {
		partition = new PartitionParhip<Topo>(FASTSOCIAL);
	}
	else if (partitioner == PartitionerType::ParHIPEcoSocial) {
		partition = new PartitionParhip<Topo>(ECOSOCIAL);
	}
#endif
	else {
		logError() << "Unknown (or disabled) partitioner.";
	}

	return std::unique_ptr<PartitionBase<Topo>>(partition);
}

	Partition() = delete;
};

using TETPartition = Partition<TETRAHEDRON>;

}

#endif // PUML_PARTITION_H
