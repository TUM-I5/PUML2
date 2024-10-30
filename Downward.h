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

#ifndef PUML_DOWNWARD_H
#define PUML_DOWNWARD_H

#include <algorithm>
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
			faceVertices(puml, cell, i, v);

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
		internal::Utils::l2g<Topo, typename PUML<Topo>::vertex_t, internal::Topology<Topo>::cellvertices()>(puml, lid, gid);
	}

	/**
	 * @param faceId The local id of the face
	 * @return The side of the cell this face is on or -1 of the face is on no side
	 */
	template<TopoType Topo>
	static int faceSide(const PUML<Topo> &puml, const typename PUML<Topo>::cell_t &cell, unsigned int faceId)
	{
		unsigned int faceIds[internal::Topology<Topo>::cellfaces()];
		faces(puml, cell, faceIds);

		unsigned int* end = faceIds+internal::Topology<Topo>::cellfaces();

		unsigned int* pFaceId = std::find(faceIds, end, faceId);
		if (pFaceId == end)
			return -1;

		return pFaceId - faceIds;
	}

	/**
	 * @param faceSide The side of the cell
	 */
	template<TopoType Topo>
	static void faceVertices(const PUML<Topo> &puml, const typename PUML<Topo>::cell_t &cell, unsigned int faceSide, unsigned int* lid)
	{
		assert(faceSide < internal::Topology<Topo>::cellfaces());
		for (unsigned int i = 0; i < internal::Topology<Topo>::facevertices(); i++) {
			lid[i] = cell.m_vertices[internal::Numbering<Topo>::facevertices()[faceSide][i]];
		}
	}
};

}

#endif // PUML_DOWNWARD_H
