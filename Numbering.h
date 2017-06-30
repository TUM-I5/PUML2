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

#ifndef PUML_NUMBERING_H
#define PUML_NUMBERING_H

#include "Topology.h"

namespace PUML
{

namespace internal
{

template<TopoType Topo>
class Numbering
{
public:
	typedef unsigned int face_t[Topology<Topo>::facevertices()];

	static const face_t* facevertices();
};

template<>
class Numbering<TETRAHEDRON>
{
public:
	typedef unsigned int face_t[Topology<TETRAHEDRON>::facevertices()];

	static const face_t* facevertices()
	{
		static const face_t vertices[4] = {
			{ 1, 0, 2 },
			{ 0, 1, 3 },
			{ 1, 2, 3 },
			{ 2, 0, 3 } };

		return vertices;
	}
};

}

}

#endif // PUML_NUMBERING_H