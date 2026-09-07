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
  using face_t = unsigned int[Topology<Topo>::facevertices()];
  using edge_t = unsigned int[2];
  using faceadj_t = unsigned int[2];

  static auto facevertices() -> const face_t*;
  static auto edgevertices() -> const edge_t*;
  static auto edgefaces() -> const faceadj_t*;
};

template <>
class Numbering<TETRAHEDRON> {
  public:
  using face_t = unsigned int[Topology<TETRAHEDRON>::facevertices()];
  using edge_t = unsigned int[2];
  using faceadj_t = unsigned int[2];

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

template <>
class Numbering<HEXAHEDRON> {
  public:
  using face_t = unsigned int[Topology<HEXAHEDRON>::facevertices()];
  using edge_t = unsigned int[2];
  using faceadj_t = unsigned int[2];

  static auto facevertices() -> const face_t* {
    static const face_t Vertices[] = {
        {0, 4, 7, 3}, {1, 2, 6, 5}, {0, 1, 5, 4}, {3, 7, 6, 2}, {0, 3, 2, 1}, {4, 5, 6, 7}};
    static_assert(std::size(Vertices) == Topology<HEXAHEDRON>::cellfaces(),
                  "The face table needs one entry per cell face");

    return Vertices;
  }

  static auto edgevertices() -> const edge_t* {
    static const edge_t Vertices[] = {{0, 1},
                                      {1, 2},
                                      {2, 3},
                                      {3, 0},
                                      {4, 5},
                                      {5, 6},
                                      {6, 7},
                                      {7, 4},
                                      {0, 4},
                                      {1, 5},
                                      {2, 6},
                                      {3, 7}};
    static_assert(std::size(Vertices) == Topology<HEXAHEDRON>::celledges(),
                  "The edge table needs one entry per cell edge");
    return Vertices;
  }

  static auto edgefaces() -> const faceadj_t* {
    static const faceadj_t Faces[] = {{2, 4},
                                      {1, 4},
                                      {3, 4},
                                      {0, 4},
                                      {2, 5},
                                      {1, 5},
                                      {3, 5},
                                      {0, 5},
                                      {0, 2},
                                      {1, 2},
                                      {1, 3},
                                      {0, 3}};
    static_assert(std::size(Faces) == Topology<HEXAHEDRON>::celledges(),
                  "The edge adjacency table needs one entry per cell edge");
    return Faces;
  }
};

template <>
class Numbering<TRIANGLE> {
  public:
  using face_t = unsigned int[Topology<TRIANGLE>::facevertices()];
  using edge_t = unsigned int[2];
  using faceadj_t = unsigned int[2];

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
  using face_t = unsigned int[Topology<QUADRANGLE>::facevertices()];
  using edge_t = unsigned int[2];
  using faceadj_t = unsigned int[2];

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
