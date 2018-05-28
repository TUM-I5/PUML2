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

#ifndef PUML_NEIGHBOR_H
#define PUML_NEIGHBOR_H

#include <cassert>

#include "Downward.h"
#include "PUML.h"
#include "Topology.h"
#include "Upward.h"

namespace PUML
{

class Neighbor
{
public:
	/**
	 * Returns all local neighor ids for a cell
	 *
	 * @param puml The PUML mesh
	 * @param clid The local id of the cell for which the neighbors should be returned
	 * @param flid The local ids of the neighbors or -1 if there is no local neighbor
	 */
	template<TopoType Topo>
	static void face(const PUML<Topo> &puml, unsigned int clid, int* flid)
	{
		unsigned int faces[internal::Topology<Topo>::cellfaces()];
		assert(clid < puml.cells().size());
		Downward::faces(puml, puml.cells()[clid], faces);

		for (unsigned int i = 0; i < internal::Topology<Topo>::cellfaces(); i++) {
			int neighbors[2];
			Upward::cells(puml, puml.faces()[faces[i]], neighbors);

			if (static_cast<unsigned int>(neighbors[0]) == clid)
				flid[i] = neighbors[1];
			else
				flid[i] = neighbors[0];
		}
	}
};

}

#endif // PUML_NEIGHBOR_H
