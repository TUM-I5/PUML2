/**
 * @file
 *  This file is part of PUML
 *
 *  For conditions of distribution and use, please see the copyright
 *  notice in the file 'COPYING' at the root directory of this package
 *  and the copyright notice at https://github.com/TUM-I5/PUMGen
 *
 * @copyright 2017 Technische Universitaet Muenchen
 * @author Sebastian Rettenberger <sebastian.rettenberger@tum.de>
 */

#ifndef PUML_DOWNWARD_H
#define PUML_DOWNWARD_H

#include <cassert>
#include <cstring>

#include "utils/arrayutils.h"

#include "Element.h"
#include "Numbering.h"
#include "PUML.h"
#include "Topology.h"
#include "Utils.h"

namespace PUML
{

class Downward
{
public:
	/**
	 * Returns all local face ids for a cell
	 *
	 * @param puml The PUML mesh
	 * @param cell The cell for which the faces should be returned
	 * @param lid The local ids of the faces
	 */
	template<TopoType Topo>
	static void faces(const PUML<Topo> &puml, const typename PUML<Topo>::cell_t &cell, unsigned int* lid)
	{
		for (unsigned int i = 0; i < internal::Topology<Topo>::cellfaces(); i++) {
			unsigned int v[internal::Topology<Topo>::facevertices()];

			for (unsigned int j = 0; j < internal::Topology<Topo>::facevertices(); j++) {
				v[j] = cell.m_vertices[internal::Numbering<Topo>::facevertices()[i][j]];
			}

			int id = puml.faceByVertices(v);
			assert(id >= 0);
			lid[i] = id;
		}
	}

	/**
	 * Returns all local vertex ids for a cell
	 *
	 * @param puml The PUML mesh
	 * @param cell The cell for which the vertices should be returned
	 * @param lid The local ids of the vertices
	 */
	template<TopoType Topo>
	static void vertices(const PUML<Topo> &puml, const typename PUML<Topo>::cell_t &cell, unsigned int* lid)
	{
		memcpy(lid, cell.m_vertices, internal::Topology<Topo>::cellvertices() * sizeof(unsigned int));
	}

	/**
	 * Returns all global vertex ids for a cell
	 *
	 * @param puml The PUML mesh
	 * @param cell The cell for which the vertices should be returned
	 * @param gid The global ids of the vertices
	 */
	template<TopoType Topo>
	static void gvertices(const PUML<Topo> &puml, const typename PUML<Topo>::cell_t &cell, unsigned long* gid)
	{
		unsigned int lid[internal::Topology<Topo>::cellvertices()];
		vertices(puml, cell, lid);
		internal::Utils::l2g<Topo, PUML<Topo>::vertex_t, internal::Topology<Topo>::cellvertices()>(puml, lid, gid);
	}
};

}

#endif // PUML_DOWNWARD_H