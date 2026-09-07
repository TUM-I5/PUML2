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
#include "Types.h"
#include <algorithm>
#include <cstring>
#include <functional>
#include <vector>

namespace PUML::internal {

/**
 * Maps a list of local vertex ids to the local id of the element they define.
 *
 * The entries sit in one open-addressed table rather than in nodes of their
 * own, so growing the map costs a handful of allocations instead of one per
 * element, and a probe stays in cache.
 */
template <unsigned int N>
class VertexElementMap {
  public:
  VertexElementMap() = default;

  auto add(const std::array<LocalId, N>& vertices) -> LocalId {
    const auto key = normalize(vertices);

    if (m_size * 4 >= m_slots.size() * 3) {
      grow();
    }

    auto& slot = m_slots[probe(key)];
    if (slot.id == InvalidLocalId) {
      slot.key = key;
      slot.id = static_cast<LocalId>(m_size);
      ++m_size;
    }
    return slot.id;
  }

  [[nodiscard]] auto find(const std::array<LocalId, N>& vertices) const -> LocalId {
    if (m_slots.empty()) {
      return InvalidLocalId;
    }
    return m_slots[probe(normalize(vertices))].id;
  }

  [[nodiscard]] auto size() const -> Size { return m_size; }

  void clear() {
    m_slots.clear();
    m_slots.shrink_to_fit();
    m_size = 0;
  }

  /// Announces how many elements are going to be added.
  void reserve(Size elements) {
    Size capacity = MinCapacity;
    while (capacity * 3 < elements * 4) {
      capacity *= 2;
    }
    if (capacity > m_slots.size()) {
      rehash(capacity);
    }
  }

  private:
  static constexpr Size MinCapacity = 1024;

  struct Slot {
    std::array<LocalId, N> key{};
    LocalId id{InvalidLocalId};
  };

  static auto normalize(const std::array<LocalId, N>& vertices) -> std::array<LocalId, N> {
    auto key = vertices;
    selectionSort<LocalId, N>(key.data());
    return key;
  }

  static auto hash(const std::array<LocalId, N>& key) -> Size {
    // Fibonacci hashing over the vertex ids, which are dense and small.
    Size h = 0;
    for (unsigned int i = 0; i < N; i++) {
      h = (h ^ static_cast<Size>(key[i])) * 0x9e3779b97f4a7c15ULL;
      h ^= h >> 29;
    }
    return h;
  }

  /// The slot holding the key, or the first free slot behind where it would be.
  [[nodiscard]] auto probe(const std::array<LocalId, N>& key) const -> Size {
    const Size mask = m_slots.size() - 1;
    Size at = hash(key) & mask;
    while (m_slots[at].id != InvalidLocalId && m_slots[at].key != key) {
      at = (at + 1) & mask;
    }
    return at;
  }

  void grow() { rehash(m_slots.empty() ? MinCapacity : m_slots.size() * 2); }

  void rehash(Size capacity) {
    std::vector<Slot> old(capacity);
    old.swap(m_slots);

    for (const auto& slot : old) {
      if (slot.id != InvalidLocalId) {
        m_slots[probe(slot.key)] = slot;
      }
    }
  }

  std::vector<Slot> m_slots;
  Size m_size{0};
};

} // namespace PUML::internal

#endif // PUML_VERTEXELEMENTMAP_H
