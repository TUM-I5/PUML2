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
template <TopoType> class Topology {
  public:
  /**
   * @return The number of vertices of a cell
   */
  static constexpr unsigned int cellvertices();

  /**
   * @return The number of faces of a cell
   */
  static constexpr unsigned int cellfaces();

  /**
   * @return The number of edges for a cell
   */
  static constexpr unsigned int celledges();

  /**
   * @return The number of edges for a face
   */
  static constexpr unsigned int faceedges() { return facevertices(); /* This is always the same */ }

  /**
   * @return The number vertices for a face
   */
  static constexpr unsigned int facevertices();
};

template <> inline constexpr unsigned int Topology<TETRAHEDRON>::cellvertices() { return 4; }

template <> inline constexpr unsigned int Topology<HEXAHEDRON>::cellvertices() { return 8; }

template <> inline constexpr unsigned int Topology<TETRAHEDRON>::cellfaces() { return 4; }

template <> inline constexpr unsigned int Topology<HEXAHEDRON>::cellfaces() { return 6; }

template <> inline constexpr unsigned int Topology<TETRAHEDRON>::celledges() { return 6; }

template <> inline constexpr unsigned int Topology<HEXAHEDRON>::celledges() { return 12; }

template <> inline constexpr unsigned int Topology<TETRAHEDRON>::facevertices() { return 3; }

template <> inline constexpr unsigned int Topology<HEXAHEDRON>::facevertices() { return 4; }

} // namespace internal

} // namespace PUML

#endif // PUML_TOPOLOGY_H