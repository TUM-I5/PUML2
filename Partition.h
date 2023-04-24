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

#ifdef USE_DKAMINPAR
#include "PartitionDKaMinPar.h"
#endif

namespace PUML
{

template<TopoType Topo>
class Partition{
public:
static std::unique_ptr<PartitionBase<Topo>> get_partitioner(const std::string& name) {
	PartitionBase<Topo>* partition = nullptr;
	if (name == "none") {
		partition = new PartitionDummy<Topo>();
	}
#ifdef USE_PARMETIS
	else if (name == "parmetis") {
		partition = new PartitionParmetis<Topo>(0);
	}
	else if (name == "parmetis-geo") {
		partition = new PartitionParmetis<Topo>(1);
	}
#endif
#ifdef USE_PTSCOTCH
	else if (name == "ptscotch") {
		partition = new PartitionPtscotch<Topo>(SCOTCH_STRATDEFAULT);
	}
	else if (name == "ptscotch-q") {
		partition = new PartitionPtscotch<Topo>(SCOTCH_STRATQUALITY);
	}
	else if (name == "ptscotch-b") {
		partition = new PartitionPtscotch<Topo>(SCOTCH_STRATBALANCE);
	}
	else if (name == "ptscotch-qb") {
		partition = new PartitionPtscotch<Topo>(SCOTCH_STRATBALANCE | SCOTCH_STRATQUALITY);
	}
	else if (name == "ptscotch-s") {
		partition = new PartitionPtscotch<Topo>(SCOTCH_STRATSPEED);
	}
	else if (name == "ptscotch-sb") {
		partition = new PartitionPtscotch<Topo>(SCOTCH_STRATBALANCE | SCOTCH_STRATSPEED);
	}
#endif
#ifdef USE_PARHIP
	else if (name == "parhip-ultrafast") {
		partition = new PartitionParhip<Topo>(ULTRAFASTMESH);
	}
	else if (name == "parhip-fast") {
		partition = new PartitionParhip<Topo>(FASTMESH);
	}
	else if (name == "parhip-eco") {
		partition = new PartitionParhip<Topo>(ECOMESH);
	}
	else if (name == "parhip-ultrafastsocial") {
		partition = new PartitionParhip<Topo>(ULTRAFASTSOCIAL);
	}
	else if (name == "parhip-fastsocial") {
		partition = new PartitionParhip<Topo>(FASTSOCIAL);
	}
	else if (name == "parhip-ecosocial") {
		partition = new PartitionParhip<Topo>(ECOSOCIAL);
	}
#endif
#ifdef USE_DKAMINPAR
	else if (name == "dkaminpar") {
		partition = new PartitionDKaMinPar<Topo>();
	}
#endif

	return std::unique_ptr<PartitionBase<Topo>>(partition);
}

	Partition() = delete;
};

using TETPartition = Partition<TETRAHEDRON>;

}

#endif // PUML_PARTITION_H
