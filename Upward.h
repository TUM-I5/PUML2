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
    std::vector<LocalId> intermediateIds;
    if constexpr (internal::Topology<Topo>::dimension() == 3) {
      edges(puml, vertex, intermediateIds);
    } else {
      faces(puml, vertex, intermediateIds);
    }

    std::vector<LocalId> cellIds;
    for (const LocalId id : intermediateIds) {
      if constexpr (internal::Topology<Topo>::dimension() == 3) {
        merge<true>(cellIds, puml.edges()[id].m_upward);
      } else {
        merge<true>(cellIds, puml.faces()[id].m_upward);
      }
    }

    if constexpr (M) {
      merge<true>(lid, cellIds);
    } else {
      std::swap(lid, cellIds);
    }
  }

  private:
  template <bool M>
  static void merge(std::vector<LocalId>& res, const std::vector<LocalId>& v);
};

template <>
inline void Upward::merge<false>(std::vector<LocalId>& res, const std::vector<LocalId>& v) {
  res = v;
}

template <>
inline void Upward::merge<true>(std::vector<LocalId>& res, const std::vector<LocalId>& v) {
  std::vector<LocalId> tmp;
  std::set_union(v.begin(), v.end(), res.begin(), res.end(), std::back_inserter(tmp));
  std::swap(tmp, res);
}

} // namespace PUML

#endif // PUML_UPWARD_H
