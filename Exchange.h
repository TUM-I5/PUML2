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

#ifndef PUML_EXCHANGE_H
#define PUML_EXCHANGE_H

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "Error.h"
#include "TypeInference.h"
#include "Types.h"

// MPI 4 counts in MPI_Count where MPI 3 counts in int. Some libraries that
// report an older version have the calls of MPI 4 all the same (Intel MPI 2021,
// for one); defining PUML_MPI_LARGE_COUNT makes PUML use them there as well,
// which the CMake build of PUML does where cmake/PumlMpiLargeCount.cmake finds
// them.
#if defined(USE_MPI) && !defined(PUML_MPI_LARGE_COUNT) && MPI_VERSION >= 4
#define PUML_MPI_LARGE_COUNT
#endif

namespace PUML::internal {

/**
 * An exchange in which every rank sends a block of entities to every rank (an
 * empty one to most): how many entities a rank sends to each rank and receives
 * from each, and where the block of each rank starts in the send and in the
 * receive buffer, which hold the blocks rank by rank.
 *
 * All of it is counted in Size. MPI 3 counts in int, which a rank with many
 * entities outgrows -- the offsets of the blocks long before any single count.
 * With PUML_MPI_LARGE_COUNT, the exchange counts in MPI_Count instead; without
 * it, an exchange that an int cannot count on any rank raises an Error on all
 * ranks rather than letting a count wrap around.
 */
class Exchange {
  public:
  Exchange() = default;

  /**
   * An exchange of which every rank knows what it sends to each rank and what
   * it receives from each.
   */
  Exchange(std::vector<Size> sendCounts, std::vector<Size> recvCounts)
      : m_sendCounts(std::move(sendCounts)), m_recvCounts(std::move(recvCounts)),
        m_sendOffsets(offsetsOf(m_sendCounts)), m_recvOffsets(offsetsOf(m_recvCounts)) {
    assert(m_sendCounts.size() == m_recvCounts.size());
  }

#ifdef USE_MPI
  /**
   * An exchange in which every rank says what it sends to each rank, and
   * learns from the others what it receives. Collective.
   */
  static auto ofSendCounts(std::vector<Size> sendCounts, MPI_Comm comm) -> Exchange {
    std::vector<Size> recvCounts(sendCounts.size());
    MPI_Alltoall(sendCounts.data(),
                 1,
                 MPITypeInfer<Size>::type(),
                 recvCounts.data(),
                 1,
                 MPITypeInfer<Size>::type(),
                 comm);
    return {std::move(sendCounts), std::move(recvCounts)};
  }
#endif // USE_MPI

  /// The same exchange the other way: what came from a rank goes back to it.
  [[nodiscard]] auto reversed() const -> Exchange {
    Exchange reverse;
    reverse.m_sendCounts = m_recvCounts;
    reverse.m_recvCounts = m_sendCounts;
    reverse.m_sendOffsets = m_recvOffsets;
    reverse.m_recvOffsets = m_sendOffsets;
#if defined(USE_MPI) && !defined(PUML_MPI_LARGE_COUNT)
    // the same values, the other way
    reverse.m_intChecked = m_intChecked;
#endif
    return reverse;
  }

  /// The number of ranks the exchange is between.
  [[nodiscard]] auto ranks() const -> int { return static_cast<int>(m_sendCounts.size()); }

  [[nodiscard]] auto sendCount(int rank) const -> Size { return m_sendCounts[rank]; }

  [[nodiscard]] auto recvCount(int rank) const -> Size { return m_recvCounts[rank]; }

  /// Where the block for a rank starts in the send buffer, in entities.
  [[nodiscard]] auto sendOffset(int rank) const -> Size { return m_sendOffsets[rank]; }

  /// Where the block from a rank starts in the receive buffer, in entities.
  [[nodiscard]] auto recvOffset(int rank) const -> Size { return m_recvOffsets[rank]; }

  /// The number of entities a rank sends, which its send buffer has to hold.
  [[nodiscard]] auto sendTotal() const -> Size { return m_sendOffsets.back(); }

  /// The number of entities a rank receives, which its receive buffer has to hold.
  [[nodiscard]] auto recvTotal() const -> Size { return m_recvOffsets.back(); }

#ifdef USE_MPI
  /**
   * Sends the blocks of `send` to their ranks and receives the blocks of the
   * other ranks into `recv`, entities of `type` each, in one all-to-all call.
   * Collective.
   */
  void alltoallv(const void* send, void* recv, MPI_Datatype type, MPI_Comm comm) const {
#ifdef PUML_MPI_LARGE_COUNT
    const std::vector<MPI_Count> sendCounts(m_sendCounts.begin(), m_sendCounts.end());
    const std::vector<MPI_Count> recvCounts(m_recvCounts.begin(), m_recvCounts.end());
    const std::vector<MPI_Aint> sendOffsets(m_sendOffsets.begin(), m_sendOffsets.end() - 1);
    const std::vector<MPI_Aint> recvOffsets(m_recvOffsets.begin(), m_recvOffsets.end() - 1);
    MPI_Alltoallv_c(send,
                    sendCounts.data(),
                    sendOffsets.data(),
                    type,
                    recv,
                    recvCounts.data(),
                    recvOffsets.data(),
                    type,
                    comm);
#else  // PUML_MPI_LARGE_COUNT
    requireIntCounts(true, comm);
    const auto sendCounts = intsOf(m_sendCounts, m_sendCounts.size());
    const auto recvCounts = intsOf(m_recvCounts, m_recvCounts.size());
    const auto sendOffsets = intsOf(m_sendOffsets, m_sendCounts.size());
    const auto recvOffsets = intsOf(m_recvOffsets, m_recvCounts.size());
    MPI_Alltoallv(send,
                  sendCounts.data(),
                  sendOffsets.data(),
                  type,
                  recv,
                  recvCounts.data(),
                  recvOffsets.data(),
                  type,
                  comm);
#endif // PUML_MPI_LARGE_COUNT
  }

  /**
   * The same exchange by a message to each rank that a block goes to, and from
   * each rank that one comes from, and no others: for an exchange in which a
   * rank deals with a few others only. Every rank has to take part.
   */
  void pointToPoint(const void* send, void* recv, MPI_Datatype type, MPI_Comm comm) const {
#ifndef PUML_MPI_LARGE_COUNT
    requireIntCounts(false, comm);
#endif // PUML_MPI_LARGE_COUNT

    MPI_Aint lowerBound = 0;
    MPI_Aint extent = 0;
    MPI_Type_get_extent(type, &lowerBound, &extent);
    const auto bytes = static_cast<Size>(extent);

    std::vector<MPI_Request> requests;
    for (int rank = 0; rank < ranks(); ++rank) {
      if (m_recvCounts[rank] > 0) {
        requests.emplace_back();
        receive(static_cast<char*>(recv) + (m_recvOffsets[rank] * bytes),
                m_recvCounts[rank],
                type,
                rank,
                comm,
                &requests.back());
      }
    }
    for (int rank = 0; rank < ranks(); ++rank) {
      if (m_sendCounts[rank] > 0) {
        requests.emplace_back();
        post(static_cast<const char*>(send) + (m_sendOffsets[rank] * bytes),
             m_sendCounts[rank],
             type,
             rank,
             comm,
             &requests.back());
      }
    }
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
  }
#endif // USE_MPI

  private:
  /// Where the block of each rank starts, and behind the last one, the total.
  static auto offsetsOf(const std::vector<Size>& counts) -> std::vector<Size> {
    std::vector<Size> offsets(counts.size() + 1);
    for (std::size_t i = 0; i < counts.size(); ++i) {
      offsets[i + 1] = offsets[i] + counts[i];
    }
    return offsets;
  }

#ifdef USE_MPI
#ifdef PUML_MPI_LARGE_COUNT
  static void post(const void* buffer,
                   Size count,
                   MPI_Datatype type,
                   int rank,
                   MPI_Comm comm,
                   MPI_Request* request) {
    MPI_Isend_c(buffer, static_cast<MPI_Count>(count), type, rank, Tag, comm, request);
  }

  static void receive(
      void* buffer, Size count, MPI_Datatype type, int rank, MPI_Comm comm, MPI_Request* request) {
    MPI_Irecv_c(buffer, static_cast<MPI_Count>(count), type, rank, Tag, comm, request);
  }
#else  // PUML_MPI_LARGE_COUNT
  /**
   * Raises an Error on all ranks if the exchange of any rank has a count, or
   * for an all-to-all call an offset, that the int MPI 3 counts in cannot
   * hold. A rank that refused on its own would leave the others waiting for it
   * in the exchange. The counts of an exchange stay as they are, so the ranks
   * agree on them once. Collective, the first time.
   */
  void requireIntCounts(bool withOffsets, MPI_Comm comm) const {
    const int needed = withOffsets ? 2 : 1;
    if (m_intChecked >= needed) {
      return;
    }

    Size largest = 0;
    for (std::size_t i = 0; i < m_sendCounts.size(); ++i) {
      largest = std::max({largest, m_sendCounts[i], m_recvCounts[i]});
      if (withOffsets) {
        largest = std::max({largest, m_sendOffsets[i], m_recvOffsets[i]});
      }
    }

    Failure failure;
    if (largest > static_cast<Size>(std::numeric_limits<int>::max())) {
      failure.report("an exchange of",
                     std::max(sendTotal(), recvTotal()),
                     "entities on a rank is more than MPI can count in an int; an MPI library"
                     " with the calls of MPI 4 can take it (with PUML_MPI_LARGE_COUNT defined"
                     " where it reports an older version)");
    }
    failure.raise(comm);
    m_intChecked = needed;
  }

  /// The first `count` values as the int that MPI 3 counts in, which they fit.
  static auto intsOf(const std::vector<Size>& values, Size count) -> std::vector<int> {
    std::vector<int> ints(count);
    for (std::size_t i = 0; i < count; ++i) {
      ints[i] = intOf(values[i]);
    }
    return ints;
  }

  static auto intOf(Size value) -> int {
    assert(value <= static_cast<Size>(std::numeric_limits<int>::max()));
    return static_cast<int>(value);
  }

  static void post(const void* buffer,
                   Size count,
                   MPI_Datatype type,
                   int rank,
                   MPI_Comm comm,
                   MPI_Request* request) {
    MPI_Isend(buffer, intOf(count), type, rank, Tag, comm, request);
  }

  static void receive(
      void* buffer, Size count, MPI_Datatype type, int rank, MPI_Comm comm, MPI_Request* request) {
    MPI_Irecv(buffer, intOf(count), type, rank, Tag, comm, request);
  }
#endif // PUML_MPI_LARGE_COUNT

  /// The tag of the messages of pointToPoint.
  static constexpr int Tag = 0;
#endif // USE_MPI

  std::vector<Size> m_sendCounts;
  std::vector<Size> m_recvCounts;
  std::vector<Size> m_sendOffsets{0};
  std::vector<Size> m_recvOffsets{0};

#if defined(USE_MPI) && !defined(PUML_MPI_LARGE_COUNT)
  /**
   * How far all ranks know that an int holds what this exchange passes to MPI:
   * 0 not yet, 1 the counts, 2 the counts and the offsets.
   */
  mutable int m_intChecked{0};
#endif
};

} // namespace PUML::internal

#endif // PUML_EXCHANGE_H
