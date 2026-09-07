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

#ifndef PUML_DATAHANDLE_H
#define PUML_DATAHANDLE_H

#include <cassert>
#include <cstddef>

#include "Topology.h"
#include "Types.h"

namespace PUML {

enum class DataType { Cell = 0, Vertex = 1 };

// some constexprs for legacy reasons
constexpr DataType CELL = DataType::Cell;
constexpr DataType VERTEX = DataType::Vertex;

template <TopoType Topo>
class PUML;

namespace internal {

/**
 * Gives every value type an address of its own, so that two types can be told
 * apart without run-time type information.
 */
template <typename T>
auto typeTag() -> const void* {
  static const char Tag = 0;
  return &Tag;
}

} // namespace internal

/**
 * The values of one data array, laid out as entities() blocks of elemCount()
 * values each.
 *
 * A view is only as good as the array it was taken from, and any call that adds
 * or replaces an array invalidates it.
 */
template <typename T>
class DataView {
  public:
  DataView() = default;

  DataView(T* values, Size entities, Size elemCount)
      : m_values(values), m_entities(entities), m_elemCount(elemCount) {}

  /// The value at a flat position, counting across all entities.
  auto operator[](Size index) const -> T& {
    assert(index < size());
    return m_values[index];
  }

  /// The values belonging to one entity.
  [[nodiscard]] auto entity(Size index) const -> T* {
    assert(index < m_entities);
    return m_values + (index * m_elemCount);
  }

  [[nodiscard]] auto data() const -> T* { return m_values; }

  [[nodiscard]] auto begin() const -> T* { return m_values; }

  [[nodiscard]] auto end() const -> T* { return m_values + size(); }

  /// The number of cells resp. vertices the array covers.
  [[nodiscard]] auto entities() const -> Size { return m_entities; }

  /// The number of values per entity.
  [[nodiscard]] auto elemCount() const -> Size { return m_elemCount; }

  /// The number of values in total.
  [[nodiscard]] auto size() const -> Size { return m_entities * m_elemCount; }

  [[nodiscard]] auto empty() const -> bool { return size() == 0; }

  private:
  T* m_values{nullptr};
  Size m_entities{0};
  Size m_elemCount{0};
};

/**
 * Names one data array of a mesh, together with the type of its values.
 *
 * A handle is obtained from the mesh that holds the array, and reading through
 * it needs no cast, so a value type cannot be mistaken for another one. It
 * stays valid while arrays are added or replaced, and it belongs to the mesh it
 * came from: handing it to a different mesh is caught in a debug build.
 */
template <typename T>
class DataHandle {
  public:
  DataHandle() = default;

  [[nodiscard]] auto valid() const -> bool { return m_elemCount > 0; }

  /// Whether the array holds cells or vertices.
  [[nodiscard]] auto type() const -> DataType { return m_type; }

  /// The number of values per entity.
  [[nodiscard]] auto elemCount() const -> Size { return m_elemCount; }

  private:
  template <TopoType Topo>
  friend class PUML;

  DataHandle(Size index, DataType type, Size elemCount, Size owner)
      : m_index(index), m_type(type), m_elemCount(elemCount), m_owner(owner) {}

  Size m_index{0};
  DataType m_type{DataType::Cell};
  Size m_elemCount{0};
  /** Checked on every use, so that a handle of another mesh cannot read here */
  Size m_owner{0};
};

} // namespace PUML

#endif // PUML_DATAHANDLE_H
