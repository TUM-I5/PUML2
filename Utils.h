// SPDX-FileCopyrightText: 2017 Technical University of Munich
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
 * @author Sebastian Rettenberger <sebastian.rettenberger@tum.de>
 */

#ifndef PUML_UTILS_H
#define PUML_UTILS_H

#include <vector>

#include "PUML.h"
#include "Topology.h"

namespace PUML
{

namespace internal
{

template<TopoType Topo, typename E>
struct ElementVector
{
	static const std::vector<E>& elements(const PUML<Topo> &puml);
};

template<TopoType Topo>
struct ElementVector<Topo, typename PUML<Topo>::cell_t>
{
	static const std::vector<typename PUML<Topo>::cell_t>& elements(const PUML<Topo> &puml)
	{ return puml.cells(); }
};

template<TopoType Topo>
struct ElementVector<Topo, typename PUML<Topo>::vertex_t>
{
	static const std::vector<typename PUML<Topo>::vertex_t>& elements(const PUML<Topo> &puml)
	{ return puml.vertices(); }
};

class Utils
{
public:
	/**
	 * Get the global ids for a set of local ids
	 */
	template<TopoType Topo, typename E, unsigned int N>
	static void l2g(const PUML<Topo> &puml, const unsigned int lid[N], unsigned long gid[N])
	{
		const std::vector<E>& elements = ElementVector<Topo, E>::elements(puml);
		for (unsigned int i = 0; i < N; i++)
			gid[i] = elements[lid[i]].m_gid;
	}
};

}

}

#endif // PUML_UTILS_H
