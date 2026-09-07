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

#ifndef PUML_VERTEXELEMENTMAP_H
#define PUML_VERTEXELEMENTMAP_H

#include "DownElement.h"
#include <algorithm>
#include <cstring>
#include <functional>
#include <unordered_map>

namespace PUML::internal {

/**
 * Mapps from a list of local vertex ids to the local id of the element
 */
template <unsigned int N>
class VertexElementMap {
  private:
  /**
   * Description of a element by the local vertex ids
   */
  struct Element {
    /** The vertices that define this element */
    std::array<unsigned int, N> vertices{};

    Element(const std::array<unsigned int, N>& vertices) : vertices(vertices) {
      selectionSort<unsigned int, N>(this->vertices.data());
    }

    auto operator==(const Element& other) const -> bool { return this->vertices == other.vertices; }
  };

  struct ElementHash {
    auto operator()(const Element& element) const -> std::size_t {
      std::size_t h = std::hash<unsigned int>{}(element.vertices[0]);
      for (unsigned int i = 1; i < N; i++) {
        hashCombine(h, element.vertices[i]);
      }

      return h;
    }
  };

  std::unordered_map<Element, unsigned int, ElementHash> m_elements;

  public:
  VertexElementMap() = default;

  auto add(const std::array<LocalId, N>& vertices) -> LocalId {
    const Element e(vertices);

    auto it = m_elements.find(e);
    if (it == m_elements.end()) {
      const auto id = static_cast<LocalId>(m_elements.size());
      it = m_elements.emplace(e, id).first;
    }

    return it->second;
  }

  [[nodiscard]] auto size() const -> size_t { return m_elements.size(); }

  auto find(const std::array<LocalId, N>& vertices) const -> LocalId {
    const auto it = m_elements.find(vertices);

    if (it == m_elements.end()) {
      return InvalidLocalId;
    }

    return it->second;
  }

  void clear() { m_elements.clear(); }

  private:
  /**
   * Taken from: https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
   */
  template <typename T>
  static void hashCombine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
};

} // namespace PUML::internal

#endif // PUML_VERTEXELEMENTMAP_H
