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

#ifndef PUML_DOWNWARD_H
#define PUML_DOWNWARD_H

#include <algorithm>
#include <cassert>
#include <cstring>

#include "CellType.h"
#include "Element.h"
#include "Numbering.h"
#include <array>
#include <utility>

#include "PUML.h"
#include "Types.h"
#include "Topology.h"
#include "Utils.h"

namespace PUML {

class Downward {
  public:
  /**
   * Returns all local face ids for a cell
   *
   * @param puml The PUML mesh
   * @param cell The cell for which the faces should be returned
   * @param lid The local ids of the faces
   */
  template <TopoType Topo>
  static void faces([[maybe_unused]] const PUML<Topo>& puml,
                    const typename PUML<Topo>::cell_t& cell,
                    LocalId* lid) {
    // a cell of a mixed mesh which has fewer faces than the widest kind leaves the rest invalid
    unsigned int faceCount = internal::Topology<Topo>::cellfaces();
    if constexpr (Topo == MIXED) {
      faceCount = internal::shapeOf(cell.type()).faceCount;
    }
    for (unsigned int i = 0; i < internal::Topology<Topo>::cellfaces(); i++) {
      if (i >= faceCount) {
        lid[i] = InvalidLocalId;
        continue;
      }
      std::array<LocalId, internal::Topology<Topo>::facevertices()> v{};
      faceVertices(puml, cell, i, v.data());

      const LocalId id = puml.faceByVertices(v);
      assert(id != InvalidLocalId);
      lid[i] = id;
    }
  }

  /**
   * Returns all local vertex ids for a cell
   *
   * @param puml The PUML mesh
   * @param cell The cell for which the vertices should be returned
   * @param lid The local ids of the vertices
   */
  template <TopoType Topo>
  static void vertices([[maybe_unused]] const PUML<Topo>& puml,
                       const typename PUML<Topo>::cell_t& cell,
                       LocalId* lid) {
    std::copy(cell.m_vertices.begin(), cell.m_vertices.end(), lid);
  }

  /**
   * Returns all global vertex ids for a cell
   *
   * @param puml The PUML mesh
   * @param cell The cell for which the vertices should be returned
   * @param gid The global ids of the vertices
   */
  template <TopoType Topo>
  static void gvertices([[maybe_unused]] const PUML<Topo>& puml,
                        const typename PUML<Topo>::cell_t& cell,
                        unsigned long* gid) {
    unsigned int lid[internal::Topology<Topo>::cellvertices()];
    vertices(puml, cell, lid);
    if constexpr (Topo == MIXED) {
      // a cell which has fewer vertices than the widest kind leaves the rest invalid
      for (unsigned int i = 0; i < internal::Topology<Topo>::cellvertices(); i++) {
        gid[i] = lid[i] == InvalidLocalId ? InvalidGlobalId : puml.vertices()[lid[i]].gid();
      }
    } else {
      internal::Utils::l2g<Topo,
                           typename PUML<Topo>::vertex_t,
                           internal::Topology<Topo>::cellvertices()>(puml, lid, gid);
    }
  }

  /**
   * @param faceId The local id of the face
   * @return The side of the cell this face is on or -1 of the face is on no side
   */
  template <TopoType Topo>
  static auto faceSide([[maybe_unused]] const PUML<Topo>& puml,
                       const typename PUML<Topo>::cell_t& cell,
                       unsigned int faceId) -> int {
    unsigned int faceIds[internal::Topology<Topo>::cellfaces()];
    faces(puml, cell, faceIds);

    unsigned int* end = faceIds + internal::Topology<Topo>::cellfaces();

    unsigned int* pFaceId = std::find(faceIds, end, faceId);
    if (pFaceId == end) {
      return -1;
    }

    return static_cast<int>(pFaceId - faceIds);
  }

  /// The faces of a cell.
  template <TopoType Topo>
  static auto faces(const PUML<Topo>& puml, const typename PUML<Topo>::cell_t& cell)
      -> std::array<LocalId, internal::Topology<Topo>::cellfaces()> {
    std::array<LocalId, internal::Topology<Topo>::cellfaces()> lid{};
    faces(puml, cell, lid.data());
    return lid;
  }

  /// The vertices of a cell.
  template <TopoType Topo>
  static auto vertices(const PUML<Topo>& puml, const typename PUML<Topo>::cell_t& cell)
      -> std::array<LocalId, internal::Topology<Topo>::cellvertices()> {
    std::array<LocalId, internal::Topology<Topo>::cellvertices()> lid{};
    vertices(puml, cell, lid.data());
    return lid;
  }

  /**
   * The vertices of a face. A mixed mesh has faces of three and of four
   * vertices, so the buffer is as wide as the wider kind and the face says how
   * much of it is used.
   */
  template <TopoType Topo>
  static auto vertices(const PUML<Topo>& puml, const typename PUML<Topo>::face_t& face)
      -> std::pair<std::array<LocalId, internal::Topology<Topo>::facevertices()>, unsigned int> {
    std::array<LocalId, internal::Topology<Topo>::facevertices()> lid{};
    lid.fill(InvalidLocalId);
    vertices(puml, face, lid.data());
    return {lid, face.vertexCount()};
  }

  /**
   * @param faceSide The side of the cell
   */
  template <TopoType Topo>
  static void faceVertices([[maybe_unused]] const PUML<Topo>& puml,
                           const typename PUML<Topo>::cell_t& cell,
                           unsigned int faceSide,
                           LocalId* lid) {
    assert(faceSide < internal::Topology<Topo>::cellfaces());
    if constexpr (Topo == MIXED) {
      // the side of a cell of one of the kinds a mixed mesh has, padded as the mesh files its faces
      const auto& shape = internal::shapeOf(cell.type());
      assert(faceSide < shape.faceCount);
      for (unsigned int i = 0; i < internal::Topology<Topo>::facevertices(); i++) {
        lid[i] = i < shape.faceVertexCount[faceSide]
                     ? cell.m_vertices[shape.faceVertices[faceSide][i]]
                     : InvalidLocalId;
      }
    } else {
      for (std::size_t i = 0; i < internal::Topology<Topo>::facevertices(); i++) {
        lid[i] = cell.m_vertices[internal::Numbering<Topo>::facevertices()[faceSide][i]];
      }
    }
  }
};

} // namespace PUML

#endif // PUML_DOWNWARD_H
