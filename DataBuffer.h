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

#ifndef PUML_DATABUFFER_H
#define PUML_DATABUFFER_H

#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

namespace PUML::internal {

/**
 * Owns the values of one data array, together with what is needed to move the
 * array between ranks.
 *
 * The values belonging to one entity are stored contiguously, so the array
 * holds entities() blocks of entitySize() bytes each.
 */
class DataBuffer {
  public:
  DataBuffer() = default;

  /**
   * @param entities The number of cells or vertices the array covers
   * @param elemCount The number of values per entity
   * @param elemBytes The size of a single value
   * @param baseType The MPI datatype of a single value
   */
  DataBuffer(std::size_t entities,
             std::size_t elemCount,
             std::size_t elemBytes
#ifdef USE_MPI
             ,
             MPI_Datatype baseType
#endif // USE_MPI
             )
      : m_values(entities * elemCount * elemBytes), m_entitySize(elemCount * elemBytes)
#ifdef USE_MPI
        ,
        m_baseType(baseType), m_elemCount(elemCount)
#endif // USE_MPI
  {
#ifdef USE_MPI
    deriveType();
#endif // USE_MPI
  }

  DataBuffer(const DataBuffer& other)
      : m_values(other.m_values), m_entitySize(other.m_entitySize)
#ifdef USE_MPI
        ,
        m_baseType(other.m_baseType), m_elemCount(other.m_elemCount)
#endif // USE_MPI
  {
#ifdef USE_MPI
    deriveType();
#endif // USE_MPI
  }

  DataBuffer(DataBuffer&& other) noexcept { swap(other); }

  /// Handles both copy and move assignment; the argument is built by whichever
  /// constructor fits the caller.
  auto operator=(DataBuffer other) noexcept -> DataBuffer& {
    swap(other);
    return *this;
  }

  ~DataBuffer() {
#ifdef USE_MPI
    freeType();
#endif // USE_MPI
  }

  void swap(DataBuffer& other) noexcept {
    m_values.swap(other.m_values);
    std::swap(m_entitySize, other.m_entitySize);
#ifdef USE_MPI
    std::swap(m_baseType, other.m_baseType);
    std::swap(m_type, other.m_type);
    std::swap(m_elemCount, other.m_elemCount);
#endif // USE_MPI
  }

  /// An array with the same value layout, holding the given number of entities.
  [[nodiscard]] auto sameLayout(std::size_t entities) const -> DataBuffer {
    DataBuffer result;
    result.m_values.resize(entities * m_entitySize);
    result.m_entitySize = m_entitySize;
#ifdef USE_MPI
    result.m_baseType = m_baseType;
    result.m_elemCount = m_elemCount;
    result.deriveType();
#endif // USE_MPI
    return result;
  }

  [[nodiscard]] auto data() -> void* { return m_values.data(); }

  [[nodiscard]] auto data() const -> const void* { return m_values.data(); }

  /// The values of one entity.
  [[nodiscard]] auto entity(std::size_t index) -> void* {
    return m_values.data() + (index * m_entitySize);
  }

  [[nodiscard]] auto entity(std::size_t index) const -> const void* {
    return m_values.data() + (index * m_entitySize);
  }

  /// Copies the values of one entity out of an array with the same layout.
  void copyEntity(std::size_t index, const DataBuffer& source, std::size_t sourceIndex) {
    std::memcpy(entity(index), source.entity(sourceIndex), m_entitySize);
  }

  /// The number of bytes per entity.
  [[nodiscard]] auto entitySize() const -> std::size_t { return m_entitySize; }

  [[nodiscard]] auto entities() const -> std::size_t {
    return m_entitySize == 0 ? 0 : m_values.size() / m_entitySize;
  }

  [[nodiscard]] auto bytes() const -> std::size_t { return m_values.size(); }

  [[nodiscard]] auto empty() const -> bool { return m_values.empty(); }

#ifdef USE_MPI
  /// The datatype describing the values of one entity.
  [[nodiscard]] auto mpiType() const -> MPI_Datatype { return m_type; }
#endif // USE_MPI

  private:
#ifdef USE_MPI
  void deriveType() {
    if (m_elemCount == 1 || m_baseType == MPI_DATATYPE_NULL) {
      m_type = m_baseType;
      return;
    }
    MPI_Type_contiguous(static_cast<int>(m_elemCount), m_baseType, &m_type);
    MPI_Type_commit(&m_type);
  }

  void freeType() {
    if (m_type != MPI_DATATYPE_NULL && m_type != m_baseType) {
      MPI_Type_free(&m_type);
    }
    m_type = MPI_DATATYPE_NULL;
  }
#endif // USE_MPI

  std::vector<std::byte> m_values;
  std::size_t m_entitySize{0};
#ifdef USE_MPI
  MPI_Datatype m_baseType{MPI_DATATYPE_NULL};
  MPI_Datatype m_type{MPI_DATATYPE_NULL};
  std::size_t m_elemCount{1};
#endif // USE_MPI
};

} // namespace PUML::internal

#endif // PUML_DATABUFFER_H
