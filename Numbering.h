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

namespace PUML::internal {

template <TopoType Topo>
class Numbering {
  public:
  typedef unsigned int face_t[Topology<Topo>::facevertices()];
  typedef unsigned int edge_t[2];
  typedef unsigned int faceadj_t[2];

  static auto facevertices() -> const face_t*;
  static auto edgevertices() -> const edge_t*;
  static auto edgefaces() -> const faceadj_t*;
};

template <>
class Numbering<TETRAHEDRON> {
  public:
  typedef unsigned int face_t[Topology<TETRAHEDRON>::facevertices()];
  typedef unsigned int edge_t[2];
  typedef unsigned int faceadj_t[2];

  static auto facevertices() -> const face_t* {
    static const face_t Vertices[4] = {{1, 0, 2}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};

    return Vertices;
  }

  static auto edgevertices() -> const edge_t* {
    static const edge_t Vertices[6] = {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}};
    return Vertices;
  }

  static auto edgefaces() -> const faceadj_t* {
    static const faceadj_t Faces[6] = {{0, 1}, {0, 2}, {0, 3}, {1, 3}, {1, 2}, {2, 3}};
    return Faces;
  }
};

// TODO(someone): adapt for HEXAHEDRON. Maybe edgefaces() can be removed or inferred

} // namespace PUML::internal

#endif // PUML_NUMBERING_H
