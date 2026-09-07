// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file
 *  This file is part of PUML
 *
 *  For conditions of distribution and use, please see the copyright
 *  notice in the file 'COPYING' at the root directory of this package
 *  and the copyright notice at https://github.com/TUM-I5/PUML
 */

#ifndef PUML_CELLTYPE_H
#define PUML_CELLTYPE_H

#include <array>
#include <cstdint>
#include <vector>

#include "Numbering.h"
#include "Topology.h"
#include "Types.h"

namespace PUML {

/**
 * The kind of a single cell, for a mesh that holds more than one kind.
 *
 * The values are the ones XDMF and VTK use, so a file can be read without a
 * translation table.
 */
enum class CellType : std::uint8_t {
  Triangle = 4,
  Quadrangle = 5,
  Tetrahedron = 6,
  Pyramid = 7,
  Wedge = 8,
  Hexahedron = 9,
};

namespace internal {

/** As wide as the widest cell kind, for the buffers a mixed mesh needs */
constexpr unsigned int MaxCellVertices = 8;
constexpr unsigned int MaxCellFaces = 6;
constexpr unsigned int MaxCellEdges = 12;
constexpr unsigned int MaxFaceVertices = 4;

/**
 * How a cell of one kind is built, in a form that can be picked at run time.
 *
 * The tables are the ones Numbering holds for that kind, widened to the widest
 * cell kind, so there is one place where they are written down.
 */
struct CellShape {
  unsigned int vertexCount{0};
  unsigned int faceCount{0};
  unsigned int edgeCount{0};
  std::array<unsigned int, MaxCellFaces> faceVertexCount{};
  std::array<std::array<unsigned int, MaxFaceVertices>, MaxCellFaces> faceVertices{};
  std::array<std::array<unsigned int, 2>, MaxCellEdges> edgeVertices{};
  std::array<std::array<unsigned int, 2>, MaxCellEdges> edgeFaces{};
};

template <TopoType Topo>
auto makeShape() -> CellShape {
  static_assert(Topology<Topo>::cellvertices() <= MaxCellVertices);
  static_assert(Topology<Topo>::cellfaces() <= MaxCellFaces);
  static_assert(Topology<Topo>::celledges() <= MaxCellEdges);
  static_assert(Topology<Topo>::facevertices() <= MaxFaceVertices);

  CellShape shape;
  shape.vertexCount = Topology<Topo>::cellvertices();
  shape.faceCount = Topology<Topo>::cellfaces();
  shape.edgeCount = Topology<Topo>::celledges();

  const auto* faces = Numbering<Topo>::facevertices();
  for (unsigned int f = 0; f < shape.faceCount; ++f) {
    shape.faceVertexCount[f] = Topology<Topo>::facevertices();
    for (unsigned int v = 0; v < Topology<Topo>::facevertices(); ++v) {
      shape.faceVertices[f][v] = faces[f][v];
    }
  }

  if constexpr (Topology<Topo>::dimension() == 3) {
    const auto* edges = Numbering<Topo>::edgevertices();
    const auto* adjacency = Numbering<Topo>::edgefaces();
    for (unsigned int e = 0; e < shape.edgeCount; ++e) {
      shape.edgeVertices[e] = {edges[e][0], edges[e][1]};
      shape.edgeFaces[e] = {adjacency[e][0], adjacency[e][1]};
    }
  }

  return shape;
}

/**
 * Builds a shape from tables written down here.
 *
 * A pyramid and a wedge only ever appear next to other kinds, so no Numbering
 * describes them and these tables are their only description. Both have faces
 * of three and of four vertices, which is why the count sits per face.
 */
inline auto makeShape(unsigned int vertexCount,
                      const std::vector<std::vector<unsigned int>>& faces,
                      const std::vector<std::array<unsigned int, 2>>& edges,
                      const std::vector<std::array<unsigned int, 2>>& edgeFaces) -> CellShape {
  CellShape shape;
  shape.vertexCount = vertexCount;
  shape.faceCount = static_cast<unsigned int>(faces.size());
  shape.edgeCount = static_cast<unsigned int>(edges.size());

  for (unsigned int f = 0; f < shape.faceCount; ++f) {
    shape.faceVertexCount[f] = static_cast<unsigned int>(faces[f].size());
    for (unsigned int v = 0; v < faces[f].size(); ++v) {
      shape.faceVertices[f][v] = faces[f][v];
    }
  }
  for (unsigned int e = 0; e < shape.edgeCount; ++e) {
    shape.edgeVertices[e] = edges[e];
    shape.edgeFaces[e] = edgeFaces[e];
  }
  return shape;
}

inline auto makePyramid() -> CellShape {
  return makeShape(5,
                   {{0, 3, 2, 1}, {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}},
                   {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 4}, {2, 4}, {3, 4}},
                   {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 4}, {1, 2}, {2, 3}, {3, 4}});
}

inline auto makeWedge() -> CellShape {
  return makeShape(6,
                   {{0, 2, 1}, {3, 4, 5}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}},
                   {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}, {0, 3}, {1, 4}, {2, 5}},
                   {{0, 2}, {0, 3}, {0, 4}, {1, 2}, {1, 3}, {1, 4}, {2, 4}, {2, 3}, {3, 4}});
}

/**
 * Describes the given kind of cell.
 */
inline auto shapeOf(CellType type) -> const CellShape& {
  static const std::array<CellShape, 4> Shapes = {
      makeShape<TETRAHEDRON>(), makePyramid(), makeWedge(), makeShape<HEXAHEDRON>()};
  switch (type) {
  case CellType::Tetrahedron:
    return Shapes[0];
  case CellType::Pyramid:
    return Shapes[1];
  case CellType::Wedge:
    return Shapes[2];
  case CellType::Hexahedron:
    return Shapes[3];
  default:
    break;
  }
  // The caller checks the kind before asking.
  return Shapes[0];
}

/**
 * Whether a mixed mesh can be built from cells of this kind.
 */
inline auto isSupported(CellType type) -> bool {
  return type == CellType::Tetrahedron || type == CellType::Pyramid || type == CellType::Wedge ||
         type == CellType::Hexahedron;
}

inline auto nameOf(CellType type) -> const char* {
  switch (type) {
  case CellType::Triangle:
    return "triangle";
  case CellType::Quadrangle:
    return "quadrangle";
  case CellType::Tetrahedron:
    return "tetrahedron";
  case CellType::Pyramid:
    return "pyramid";
  case CellType::Wedge:
    return "wedge";
  case CellType::Hexahedron:
    return "hexahedron";
  }
  return "unknown";
}

} // namespace internal

} // namespace PUML

#endif // PUML_CELLTYPE_H
