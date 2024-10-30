// SPDX-FileCopyrightText: 2017-2024 Technical University of Munich
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
 * @author David Schneller <david.schneller@tum.de>
 */

#ifndef PUML_NUMBERING_H
#define PUML_NUMBERING_H

#include "Topology.h"

namespace PUML {

namespace internal {

template <TopoType Topo>
class Numbering {
  public:
  typedef unsigned int face_t[Topology<Topo>::facevertices()];
  typedef unsigned int edge_t[2];
  typedef unsigned int faceadj_t[2];

  static const face_t* facevertices();
  static const edge_t* edgevertices();
  static const faceadj_t* edgefaces();
};

template <>
class Numbering<TETRAHEDRON> {
  public:
  typedef unsigned int face_t[Topology<TETRAHEDRON>::facevertices()];
  typedef unsigned int edge_t[2];
  typedef unsigned int faceadj_t[2];

  static const face_t* facevertices() {
    static const face_t vertices[4] = {{1, 0, 2}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};

    return vertices;
  }

  static const edge_t* edgevertices() {
    static const edge_t vertices[6] = {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}};
    return vertices;
  }

  static const faceadj_t* edgefaces() {
    static const faceadj_t faces[6] = {{0, 1}, {0, 2}, {0, 3}, {1, 3}, {1, 2}, {2, 3}};
    return faces;
  }
};

// TODO(someone): adapt for HEXAHEDRON. Maybe edgefaces() can be removed or inferred

} // namespace internal

} // namespace PUML

#endif // PUML_NUMBERING_H
