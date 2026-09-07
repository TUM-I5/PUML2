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

#include <iterator>

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
    static const face_t Vertices[] = {{1, 0, 2}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
    static_assert(std::size(Vertices) == Topology<TETRAHEDRON>::cellfaces(),
                  "The face table needs one entry per cell face");

    return Vertices;
  }

  static auto edgevertices() -> const edge_t* {
    static const edge_t Vertices[] = {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}};
    static_assert(std::size(Vertices) == Topology<TETRAHEDRON>::celledges(),
                  "The edge table needs one entry per cell edge");
    return Vertices;
  }

  static auto edgefaces() -> const faceadj_t* {
    static const faceadj_t Faces[] = {{0, 1}, {0, 2}, {0, 3}, {1, 3}, {1, 2}, {2, 3}};
    static_assert(std::size(Faces) == Topology<TETRAHEDRON>::celledges(),
                  "The edge adjacency table needs one entry per cell edge");
    return Faces;
  }
};

// TODO(someone): adapt for HEXAHEDRON. Also, maybe edgefaces() can be removed or inferred

template <>
class Numbering<TRIANGLE> {
  public:
  typedef unsigned int face_t[Topology<TRIANGLE>::facevertices()];
  typedef unsigned int edge_t[2];
  typedef unsigned int faceadj_t[2];

  static auto facevertices() -> const face_t* {
    static const face_t Vertices[] = {{1, 0}, {2, 1}, {0, 2}};
    static_assert(std::size(Vertices) == Topology<TRIANGLE>::cellfaces(),
                  "The face table needs one entry per cell face");

    return Vertices;
  }

  static auto edgevertices() -> const edge_t* {
    // not needed for 2D
    return nullptr;
  }

  static auto edgefaces() -> const faceadj_t* {
    // not needed for 2D (conflates with the faces)
    return nullptr;
  }
};

template <>
class Numbering<QUADRANGLE> {
  public:
  typedef unsigned int face_t[Topology<QUADRANGLE>::facevertices()];
  typedef unsigned int edge_t[2];
  typedef unsigned int faceadj_t[2];

  static auto facevertices() -> const face_t* {
    static const face_t Vertices[] = {{1, 0}, {2, 1}, {3, 2}, {0, 3}};
    static_assert(std::size(Vertices) == Topology<QUADRANGLE>::cellfaces(),
                  "The face table needs one entry per cell face");

    return Vertices;
  }

  static auto edgevertices() -> const edge_t* {
    // not needed for 2D
    return nullptr;
  }

  static auto edgefaces() -> const faceadj_t* {
    // not needed for 2D (conflates with the faces)
    return nullptr;
  }
};

} // namespace PUML::internal

#endif // PUML_NUMBERING_H
