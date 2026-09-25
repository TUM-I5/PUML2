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

#ifndef PUML_UPWARD_H
#define PUML_UPWARD_H

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>
#include <iterator>

#include <array>

#include "PUML.h"
#include "Types.h"
#include "Topology.h"

namespace PUML {

class Upward {
  public:
  /**
   * Returns all local cell ids for a face
   *
   * @param puml The PUML mesh
   * @param cell The face for which the cells should be returned
   * @param lid The local ids of the cells
   */
  template <TopoType Topo>
  static void
      cells(const PUML<Topo>& /*puml*/, const typename PUML<Topo>::face_t& face, LocalId* lid) {
    std::copy(face.m_upward.begin(), face.m_upward.end(), lid);
  }

  template <TopoType Topo, bool M = false>
  static void faces(const PUML<Topo>& /*puml*/,
                    const typename PUML<Topo>::edge_t& edge,
                    std::vector<LocalId>& lid) {
    merge<M>(lid, edge.m_upward);
  }

  template <TopoType Topo, bool M = false>
  static void cells([[maybe_unused]] const PUML<Topo>& puml,
                    const typename PUML<Topo>::edge_t& edge,
                    std::vector<LocalId>& lid) {
    std::vector<LocalId> faceIds;
    faces(puml, edge, faceIds);

    std::vector<LocalId> cellIds;
    for (const LocalId faceId : faceIds) {
      LocalId tmp[2];
      cells(puml, puml.faces()[faceId], tmp);
      const unsigned int c = (tmp[1] == InvalidLocalId ? 1 : 2);

      std::vector<LocalId> merged;
      std::set_union(cellIds.begin(), cellIds.end(), tmp, tmp + c, std::back_inserter(merged));
      std::swap(merged, cellIds);
    }

    if constexpr (M) {
      merge<true>(lid, cellIds);
    } else {
      std::swap(lid, cellIds);
    }
  }

  template <TopoType Topo, bool M = false>
  static void edges(const PUML<Topo>& /*puml*/,
                    const typename PUML<Topo>::vertex_t& vertex,
                    std::vector<LocalId>& lid) {
    merge<M>(lid, vertex.m_upward);
  }

  template <TopoType Topo, bool M = false>
  static void cells([[maybe_unused]] const PUML<Topo>& puml,
                    const typename PUML<Topo>::vertex_t& vertex,
                    std::vector<LocalId>& lid) {
    // In three dimensions a vertex knows its edges, and an edge its faces. In
    // two there are no edges, and a vertex knows its faces directly.
    std::vector<LocalId> intermediateIds;
    merge<false>(intermediateIds, vertex.m_upward);

    std::vector<LocalId> cellIds;
    for (const LocalId id : intermediateIds) {
      if constexpr (internal::Topology<Topo>::dimension() == 3) {
        merge<true>(cellIds, puml.edges()[id].m_upward);
      } else {
        std::array<LocalId, 2> adjacent{};
        cells(puml, puml.faces()[id], adjacent.data());
        std::vector<LocalId> present;
        for (const auto cell : adjacent) {
          if (cell != InvalidLocalId) {
            present.push_back(cell);
          }
        }
        merge<true>(cellIds, present);
      }
    }

    if constexpr (M) {
      merge<true>(lid, cellIds);
    } else {
      std::swap(lid, cellIds);
    }
  }

  /// The cells a face belongs to; the second is InvalidLocalId on a boundary
  /// face or where the neighbour is on another rank.
  template <TopoType Topo>
  static auto cells(const PUML<Topo>& puml, const typename PUML<Topo>::face_t& face)
      -> std::array<LocalId, 2> {
    std::array<LocalId, 2> lid{};
    cells(puml, face, lid.data());
    return lid;
  }

  /// The faces an edge belongs to.
  template <TopoType Topo>
  static auto faces(const PUML<Topo>& puml, const typename PUML<Topo>::edge_t& edge)
      -> std::vector<LocalId> {
    std::vector<LocalId> lid;
    faces(puml, edge, lid);
    return lid;
  }

  /// The edges a vertex belongs to.
  template <TopoType Topo>
  static auto edges(const PUML<Topo>& puml, const typename PUML<Topo>::vertex_t& vertex)
      -> std::vector<LocalId> {
    std::vector<LocalId> lid;
    edges(puml, vertex, lid);
    return lid;
  }

  private:
  template <bool M, typename UpwardT>
  static void merge(std::vector<LocalId>& res, const UpwardT& v);
};

template <bool M, typename UpwardT>
inline void Upward::merge(std::vector<LocalId>& res, const UpwardT& v) {
  if constexpr (M) {
    std::vector<LocalId> merged;
    std::set_union(res.begin(), res.end(), v.begin(), v.end(), std::back_inserter(merged));
    std::swap(merged, res);
  } else {
    res.assign(v.begin(), v.end());
  }
}
} // namespace PUML

#endif // PUML_UPWARD_H
