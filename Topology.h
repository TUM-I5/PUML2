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
enum TopoType { TRIANGLE, QUADRANGLE, TETRAHEDRON, HEXAHEDRON };

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

  /**
   * @return The spatial dimension of the simplex (either 2D or 3D).
   */
  static constexpr auto dimension() -> unsigned int;
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

template <>
constexpr auto Topology<TETRAHEDRON>::dimension() -> unsigned int {
  return 3;
}

template <>
constexpr auto Topology<HEXAHEDRON>::dimension() -> unsigned int {
  return 3;
}

template <>
constexpr auto Topology<TRIANGLE>::cellvertices() -> unsigned int {
  return 3;
}

template <>
constexpr auto Topology<QUADRANGLE>::cellvertices() -> unsigned int {
  return 4;
}

template <>
constexpr auto Topology<TRIANGLE>::cellfaces() -> unsigned int {
  return 4;
}

template <>
constexpr auto Topology<QUADRANGLE>::cellfaces() -> unsigned int {
  return 4;
}

template <>
constexpr auto Topology<TRIANGLE>::celledges() -> unsigned int {
  // 2D
  return 0;
}

template <>
constexpr auto Topology<QUADRANGLE>::celledges() -> unsigned int {
  // 2D
  return 0;
}

template <>
constexpr auto Topology<TRIANGLE>::facevertices() -> unsigned int {
  // 2D
  return 2;
}

template <>
constexpr auto Topology<QUADRANGLE>::facevertices() -> unsigned int {
  // 2D
  return 2;
}

template <>
constexpr auto Topology<TRIANGLE>::dimension() -> unsigned int {
  return 2;
}

template <>
constexpr auto Topology<QUADRANGLE>::dimension() -> unsigned int {
  return 2;
}

} // namespace internal

} // namespace PUML

#endif // PUML_TOPOLOGY_H
