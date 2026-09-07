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

#ifndef PUML_RAGGEDBUFFER_H
#define PUML_RAGGEDBUFFER_H

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

#include "DataBuffer.h"
#include "DataHandle.h"
#include "Types.h"

namespace PUML {

/**
 * The values of a data array that holds a different number of them per entity.
 *
 * The values of all entities lie in one array, and the offsets say where the
 * ones of each entity begin.
 */
template <typename T>
class RaggedView {
  public:
  RaggedView() = default;

  RaggedView(T* values, const Size* offsets, Size entities)
      : m_values(values), m_offsets(offsets), m_entities(entities) {}

  [[nodiscard]] auto begin(Size entity) const -> T* {
    assert(entity < m_entities);
    return m_values + m_offsets[entity];
  }

  [[nodiscard]] auto end(Size entity) const -> T* {
    assert(entity < m_entities);
    return m_values + m_offsets[entity + 1];
  }

  /// The number of values of one entity.
  [[nodiscard]] auto count(Size entity) const -> Size {
    assert(entity < m_entities);
    return m_offsets[entity + 1] - m_offsets[entity];
  }

  /// The values of all entities, one after the other.
  [[nodiscard]] auto data() const -> T* { return m_values; }

  [[nodiscard]] auto entities() const -> Size { return m_entities; }

  /// The number of values in total.
  [[nodiscard]] auto size() const -> Size {
    return m_entities == 0 ? 0 : m_offsets[m_entities] - m_offsets[0];
  }

  private:
  T* m_values{nullptr};
  const Size* m_offsets{nullptr};
  Size m_entities{0};
};

/**
 * Names a data array that holds a different number of values per entity,
 * together with the type of its values. See DataHandle.
 */
template <typename T>
class RaggedHandle {
  public:
  RaggedHandle() = default;

  [[nodiscard]] auto valid() const -> bool { return m_valid; }

  [[nodiscard]] auto type() const -> DataType { return m_type; }

  private:
  template <TopoType Topo>
  friend class PUML;

  RaggedHandle(Size index, DataType type, [[maybe_unused]] Size owner)
      : m_index(index), m_type(type), m_valid(true)
#ifndef NDEBUG
        ,
        m_owner(owner)
#endif // NDEBUG
  {
  }

  Size m_index{0};
  DataType m_type{DataType::Cell};
  bool m_valid{false};
#ifndef NDEBUG
  Size m_owner{0};
#endif // NDEBUG
};

namespace internal {

/**
 * Owns the values of one data array that holds a different number of them per
 * entity, together with the offsets that split them up.
 *
 * The values sit in a DataBuffer of one value per entry, so they travel between
 * ranks the same way a rectangular array does; the offsets say how many of them
 * belong to each entity.
 */
class RaggedBuffer {
  public:
  RaggedBuffer() = default;

  RaggedBuffer(std::vector<Size> offsets, DataBuffer values)
      : m_offsets(std::move(offsets)), m_values(std::move(values)) {
    assert(!m_offsets.empty());
    assert(m_offsets.back() == m_values.entities());
  }

  [[nodiscard]] auto entities() const -> Size {
    return m_offsets.empty() ? 0 : m_offsets.size() - 1;
  }

  [[nodiscard]] auto count(Size entity) const -> Size {
    return m_offsets[entity + 1] - m_offsets[entity];
  }

  [[nodiscard]] auto offsets() const -> const std::vector<Size>& { return m_offsets; }

  [[nodiscard]] auto values() -> DataBuffer& { return m_values; }

  [[nodiscard]] auto values() const -> const DataBuffer& { return m_values; }

  /// Builds the offsets of an array holding the given number of values per
  /// entity.
  static auto offsetsOf(const std::vector<Size>& counts) -> std::vector<Size> {
    std::vector<Size> offsets(counts.size() + 1);
    for (Size i = 0; i < counts.size(); ++i) {
      offsets[i + 1] = offsets[i] + counts[i];
    }
    return offsets;
  }

  /// Replaces the content, keeping the value layout.
  void reset(std::vector<Size> offsets, DataBuffer values) {
    m_offsets = std::move(offsets);
    m_values = std::move(values);
  }

  private:
  /** entities()+1 entries, ascending */
  std::vector<Size> m_offsets;
  DataBuffer m_values;
};

} // namespace internal

} // namespace PUML

#endif // PUML_RAGGEDBUFFER_H
