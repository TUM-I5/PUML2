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

#ifndef PUML_ELEMENT_H
#define PUML_ELEMENT_H

#include <vector>

#include "Topology.h"

namespace PUML
{

namespace internal
{

class Utils;

} // namespace internal

template<TopoType Topo>
class PUML;

class Downward;

template<typename utype>
class Element
{
	template<TopoType Topo>
	friend class PUML;

	friend class Downward;

	friend class internal::Utils;

private:
	/** The global id */
	unsigned long m_gid;

	/** The local/global ids of the upper elements */
	utype m_upward;

public:
	/**
	 * @return The global ID of the element
	 */
	unsigned long gid() const
	{ return m_gid; }
};

/**
 * Elements that can be on partition boundaries (all except cells)
 */
template<typename utype>
class BoundaryElement : public Element<utype>
{
	template<TopoType Topo>
	friend class PUML;

private:
	/** A listof ranks that contain the same vertex */
	std::vector<int> m_sharedRanks;

public:
	/**
	 * @return <code>True</code> if the element is on a partition boundary
	 */
	bool isShared()
	{ return m_sharedRanks.size() > 1; }
};

class Vertex : public BoundaryElement<std::vector<unsigned int> >
{
	template<TopoType Topo>
	friend class PUML;

private:
	double m_coordinate[3];

public:
	/**
	 * @return A pointer to an array with 3 components containing
	 *  x, y and z
	 */
	const double* coordinate() const
	{ return m_coordinate; }
};

class Edge : public BoundaryElement<std::vector<unsigned int> >
{
	template<TopoType Topo>
	friend class PUML;
};

class Face : public BoundaryElement<unsigned int[2]>
{
	template<TopoType Topo>
	friend class PUML;
};

template<TopoType Topo>
class Cell : public Element<unsigned int[0]>
{
	friend class PUML<Topo>;

public:
	unsigned int m_vertices[internal::Topology<Topo>::cellvertices()];
};

}

#endif // PUML_ELEMENT_H