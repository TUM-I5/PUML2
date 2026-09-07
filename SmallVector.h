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

#ifndef PUML_SMALLVECTOR_H
#define PUML_SMALLVECTOR_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <type_traits>

#include "Types.h"

namespace PUML::internal {

/**
 * A sequence of values that keeps the first Capacity of them next to the
 * object and only reaches for the heap beyond that.
 *
 * Most entities of a mesh have a handful of neighbours, so a container of their
 * own would mean one small allocation per entity. Capacity should be picked so
 * that the usual case fits.
 */
template <typename T, unsigned int Capacity>
class SmallVector {
  static_assert(std::is_trivially_copyable_v<T>, "T needs to be trivially copyable");
  static_assert(Capacity > 0, "Capacity needs to leave room for at least one value");

  public:
  using value_type = T;

  SmallVector() = default;

  SmallVector(const SmallVector& other) { assign(other.begin(), other.end()); }

  SmallVector(SmallVector&& other) noexcept { steal(other); }

  auto operator=(const SmallVector& other) -> SmallVector& {
    if (this != &other) {
      assign(other.begin(), other.end());
    }
    return *this;
  }

  auto operator=(SmallVector&& other) noexcept -> SmallVector& {
    if (this != &other) {
      release();
      steal(other);
    }
    return *this;
  }

  ~SmallVector() { release(); }

  void assign(const T* first, const T* last) {
    const auto count = static_cast<Size>(last - first);
    reserve(count);
    if (count > 0) {
      std::memcpy(data(), first, count * sizeof(T));
    }
    m_size = static_cast<LocalId>(count);
  }

  void push_back(const T& value) {
    if (m_size == m_capacity) {
      reserve(static_cast<Size>(m_capacity) * 2);
    }
    data()[m_size] = value;
    ++m_size;
  }

  void reserve(Size count) {
    if (count <= m_capacity) {
      return;
    }
    T* fresh = new T[count];
    if (m_size > 0) {
      std::memcpy(fresh, data(), static_cast<Size>(m_size) * sizeof(T));
    }
    release();
    m_heap = fresh;
    m_capacity = static_cast<LocalId>(count);
  }

  [[nodiscard]] auto data() -> T* { return onHeap() ? m_heap : m_inline; }

  [[nodiscard]] auto data() const -> const T* { return onHeap() ? m_heap : m_inline; }

  auto operator[](Size index) -> T& {
    assert(index < m_size);
    return data()[index];
  }

  auto operator[](Size index) const -> const T& {
    assert(index < m_size);
    return data()[index];
  }

  [[nodiscard]] auto begin() -> T* { return data(); }
  [[nodiscard]] auto end() -> T* { return data() + m_size; }
  [[nodiscard]] auto begin() const -> const T* { return data(); }
  [[nodiscard]] auto end() const -> const T* { return data() + m_size; }

  [[nodiscard]] auto size() const -> Size { return m_size; }
  [[nodiscard]] auto empty() const -> bool { return m_size == 0; }

  void resize(Size count) {
    reserve(count);
    if (count > m_size) {
      std::fill(data() + m_size, data() + count, T{});
    }
    m_size = static_cast<LocalId>(count);
  }

  void clear() { m_size = 0; }

  auto operator==(const SmallVector& other) const -> bool {
    return m_size == other.m_size && std::equal(begin(), end(), other.begin());
  }

  private:
  [[nodiscard]] auto onHeap() const -> bool { return m_capacity > Capacity; }

  void release() {
    if (onHeap()) {
      delete[] m_heap;
    }
    m_heap = nullptr;
    m_capacity = Capacity;
    m_size = 0;
  }

  void steal(SmallVector& other) {
    if (other.onHeap()) {
      m_heap = other.m_heap;
      m_capacity = other.m_capacity;
      m_size = other.m_size;
      other.m_heap = nullptr;
      other.m_capacity = Capacity;
      other.m_size = 0;
    } else {
      assign(other.begin(), other.end());
      other.m_size = 0;
    }
  }

  union {
    T m_inline[Capacity];
    T* m_heap;
  };
  LocalId m_size{0};
  LocalId m_capacity{Capacity};
};

} // namespace PUML::internal

#endif // PUML_SMALLVECTOR_H
