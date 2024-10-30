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

#ifndef PUML_TOPOLOGY_H
#define PUML_TOPOLOGY_H

namespace PUML {

/**
 * The topology types
 */
enum TopoType { TETRAHEDRON, HEXAHEDRON };

namespace internal {

/**
 * Class describing the different topologies
 */
template <TopoType>
class Topology {
  public:
  /**
   * @return The number of vertices of a cell
   */
  static constexpr auto cellvertices() -> unsigned int;

  /**
   * @return The number of faces of a cell
   */
  static constexpr auto cellfaces() -> unsigned int;

  /**
   * @return The number of edges for a cell
   */
  static constexpr auto celledges() -> unsigned int;

  /**
   * @return The number of edges for a face
   */
  static constexpr auto faceedges() -> unsigned int {
    return facevertices(); /* This is always the same */
  }

  /**
   * @return The number vertices for a face
   */
  static constexpr auto facevertices() -> unsigned int;
};

template <>
constexpr auto Topology<TETRAHEDRON>::cellvertices() -> unsigned int {
  return 4;
}

template <>
constexpr auto Topology<HEXAHEDRON>::cellvertices() -> unsigned int {
  return 8;
}

template <>
constexpr auto Topology<TETRAHEDRON>::cellfaces() -> unsigned int {
  return 4;
}

template <>
constexpr auto Topology<HEXAHEDRON>::cellfaces() -> unsigned int {
  return 6;
}

template <>
constexpr auto Topology<TETRAHEDRON>::celledges() -> unsigned int {
  return 6;
}

template <>
constexpr auto Topology<HEXAHEDRON>::celledges() -> unsigned int {
  return 12;
}

template <>
constexpr auto Topology<TETRAHEDRON>::facevertices() -> unsigned int {
  return 3;
}

template <>
constexpr auto Topology<HEXAHEDRON>::facevertices() -> unsigned int {
  return 4;
}

} // namespace internal

} // namespace PUML

#endif // PUML_TOPOLOGY_H
