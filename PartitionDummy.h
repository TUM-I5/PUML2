// SPDX-FileCopyrightText: 2019-2023 Technical University of Munich
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
 * @copyright 2019-2023 Technische Universitaet Muenchen
 * @author David Schneller <david.schneller@tum.de>
 */

#ifndef PUML_PARTITIONDUMMY_H
#define PUML_PARTITIONDUMMY_H

#include "PartitionTarget.h"
#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include "PartitionBase.h"
#include "PartitionGraph.h"
#include "Topology.h"

namespace PUML {

template <TopoType Topo>
class PartitionDummy : public PartitionBase<Topo> {

  public:
  using PartitionBase<Topo>::PartitionBase;

  auto partition(int* partition,
                 const PartitionGraph<Topo>& graph,
                 const PartitionTarget& /*target*/,
                 int /*seed*/ = 1) -> PartitioningResult override {
    // all data stays where it is (i.e. where it was read); without MPI, that is
    // the only rank there is
    int rank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(graph.comm(), &rank);
#endif // USE_MPI
    for (unsigned long i = 0; i < graph.localVertexCount(); ++i) {
      partition[i] = rank;
    }
    return PartitioningResult::SUCCESS;
  }
};

} // namespace PUML

#endif // PUML_PARTITIONPARMETIS_H
