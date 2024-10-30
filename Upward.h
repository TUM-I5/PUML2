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

#include "PUML.h"
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
  static void cells(const PUML<Topo>& puml, const typename PUML<Topo>::face_t& face, int* lid) {
    memcpy(lid, face.m_upward, 2 * sizeof(int));
  }

  template <TopoType Topo, bool M = false>
  static void faces(const PUML<Topo>& puml,
                    const typename PUML<Topo>::edge_t& edge,
                    std::vector<int>& lid) {
    merge<M>(lid, edge.m_upward);
  }

  template <TopoType Topo, bool M = false>
  static void cells(const PUML<Topo>& puml,
                    const typename PUML<Topo>::edge_t& edge,
                    std::vector<int>& lid) {
    std::vector<int> faceIds;
    faces(puml, edge, faceIds);

    std::vector<int> cellIds;
    for (std::vector<int>::const_iterator it = faceIds.begin(); it != faceIds.end(); ++it) {
      int tmp[2];
      cells(puml, puml.faces()[*it], tmp);
      unsigned int c = (tmp[1] < 0 ? 1 : 2);

      std::vector<int> merged;
      std::set_union(cellIds.begin(), cellIds.end(), tmp, tmp + c, std::back_inserter(merged));
      std::swap(merged, cellIds);
    }

    if (M)
      merge<true>(lid, cellIds);
    else
      std::swap(lid, cellIds);
  }

  template <TopoType Topo, bool M = false>
  static void edges(const PUML<Topo>& puml,
                    const typename PUML<Topo>::vertex_t& vertex,
                    std::vector<int>& lid) {
    merge<M>(lid, vertex.m_upward);
  }

  template <TopoType Topo, bool M = false>
  static void cells(const PUML<Topo>& puml,
                    const typename PUML<Topo>::vertex_t& vertex,
                    std::vector<int>& lid) {
    std::vector<int> edgeIds;
    edges(puml, vertex, edgeIds);

    std::vector<int> cellIds;
    for (std::vector<int>::const_iterator it = edgeIds.begin(); it != edgeIds.end(); ++it) {
      merge<true>(cellIds, puml.edges()[*it].m_upward);
    }

    if (M)
      merge<true>(lid, cellIds);
    else
      std::swap(lid, cellIds);
  }

  private:
  template <bool M>
  static void merge(std::vector<int>& res, const std::vector<int>& v);
};

template <>
inline void Upward::merge<false>(std::vector<int>& res, const std::vector<int>& v) {
  res = v;
}

template <>
inline void Upward::merge<true>(std::vector<int>& res, const std::vector<int>& v) {
  std::vector<int> tmp;
  std::set_union(v.begin(), v.end(), res.begin(), res.end(), std::back_inserter(tmp));
  std::swap(tmp, res);
}

} // namespace PUML

#endif // PUML_UPWARD_H
