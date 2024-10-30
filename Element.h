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

#ifndef PUML_ELEMENT_H
#define PUML_ELEMENT_H

#include <vector>

#include "Topology.h"

namespace PUML {

namespace internal {

class Utils;

} // namespace internal

template <TopoType Topo>
class PUML;
class Downward;
class Upward;

template <typename utype>
class Element {
  template <TopoType Topo>
  friend class PUML;
  friend class Upward;
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
  unsigned long gid() const { return m_gid; }
};

/**
 * Elements that can be on partition boundaries (all except cells)
 */
template <typename utype>
class BoundaryElement : public Element<utype> {
  template <TopoType Topo>
  friend class PUML;

  private:
  /** A listof ranks that contain the same vertex */
  std::vector<int> m_sharedRanks;

  public:
  /**
   * @return <code>True</code> if the element is on a partition boundary
   */
  bool isShared() const { return m_sharedRanks.size() > 0; }

  /**
   * @return A vector of ranks that also has this element
   */
  const std::vector<int>& shared() const { return m_sharedRanks; }
};

class Vertex : public BoundaryElement<std::vector<int>> {
  template <TopoType Topo>
  friend class PUML;

  private:
  double m_coordinate[3];

  public:
  /**
   * @return A pointer to an array with 3 components containing
   *  x, y and z
   */
  const double* coordinate() const { return m_coordinate; }
};

class Edge : public BoundaryElement<std::vector<int>> {
  template <TopoType Topo>
  friend class PUML;
};

class Face : public BoundaryElement<int[2]> {
  template <TopoType Topo>
  friend class PUML;
};

template <TopoType Topo>
class Cell : public Element<std::array<int, 0>> {
  friend class PUML<Topo>;
  friend class Downward;

  private:
  unsigned int m_vertices[internal::Topology<Topo>::cellvertices()];
};

} // namespace PUML

#endif // PUML_ELEMENT_H