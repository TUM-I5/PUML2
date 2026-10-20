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

#ifndef PUML_ERROR_H
#define PUML_ERROR_H

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace PUML {

/**
 * Reports that the mesh, or the way it is being used, does not hold up.
 *
 * A caller that has no way to recover can let it escape, which ends the program
 * the way an aborting error did. One that wants to report the problem itself,
 * or a test that wants to provoke it, can catch it.
 *
 * Under MPI, a defect that the construction of the mesh finds is raised on all
 * ranks of the communicator together, with the same message, even when a
 * single rank holds the part of the mesh that has it; no rank is left waiting
 * for another one that has stopped. A misuse of the interface is raised where
 * it happens, which is on all ranks alike when all of them make the same calls.
 */
class Error : public std::runtime_error {
  public:
  explicit Error(const std::string& what) : std::runtime_error(what) {}
};

namespace internal {

/**
 * Whether a piece of a message is a punctuation mark that belongs to the piece
 * before it: the mark alone, or the mark and a space ahead of more words. A
 * piece such as a relative path, which starts with a dot, is none.
 */
inline auto followsDirectly(const std::string& piece) -> bool {
  return !piece.empty() &&
         std::string_view(",.;:)").find(piece.front()) != std::string_view::npos &&
         (piece.size() == 1 || piece[1] == ' ');
}

/**
 * Builds the message of an Error from the pieces handed to it, separated by
 * spaces. A punctuation mark follows the piece before it directly.
 */
inline void formatError(std::ostringstream& out) { static_cast<void>(out); }

template <typename T, typename... Rest>
void formatError(std::ostringstream& out, const T& first, const Rest&... rest) {
  std::ostringstream piece;
  piece << first;
  const auto text = piece.str();
  if (!text.empty()) {
    if (!followsDirectly(text)) {
      out << ' ';
    }
    out << text;
  }
  formatError(out, rest...);
}

/**
 * The values of a range as one piece of a message, in parentheses and
 * separated by commas.
 */
template <typename It>
auto listOf(It first, It last) -> std::string {
  std::ostringstream out;
  out << '(';
  for (auto it = first; it != last; ++it) {
    if (it != first) {
      out << ", ";
    }
    // a value of a character type is a number here
    out << +*it;
  }
  out << ')';
  return out.str();
}

} // namespace internal

/**
 * Raises an Error whose message is the pieces handed over, separated by spaces.
 */
template <typename... Args>
[[noreturn]] void throwError(const Args&... args) {
  std::ostringstream out;
  out << "PUML:";
  internal::formatError(out, args...);
  throw Error(out.str());
}

namespace internal {

/**
 * The first defect that this rank has found in the mesh, kept until all ranks
 * can raise it together.
 *
 * A rank sees its own part of the mesh only. Were it to raise a defect there on
 * its own, the other ranks would go on into the next exchange and wait for it
 * there forever. A check reports to a Failure instead, and all ranks raise it
 * at a point that every one of them passes.
 */
class Failure {
  public:
  /**
   * Reports a defect, with a message made of the pieces handed over as
   * throwError makes it. Only the first defect a rank reports is kept.
   */
  template <typename... Args>
  void report(const Args&... args) {
    if (!failed()) {
      std::ostringstream out;
      out << "PUML:";
      formatError(out, args...);
      m_message = out.str();
    }
  }

  /**
   * Keeps an error that was raised on this rank alone, unless a defect is kept
   * already.
   */
  void record(const Error& error) {
    if (!failed()) {
      m_message = error.what();
    }
  }

  /**
   * Whether this rank has found a defect.
   */
  [[nodiscard]] auto failed() const -> bool { return !m_message.empty(); }

#ifdef USE_MPI
  /**
   * Raises the defect on all ranks of the communicator if any rank has found
   * one, and returns if none has. All ranks raise the message of the lowest
   * rank with a defect, together with that rank and how many more ranks found
   * one. Collective.
   */
  void raise(MPI_Comm comm) const {
    int failures = failed() ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_SUM, comm);
    if (failures == 0) {
      return;
    }

    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    int first = failed() ? rank : std::numeric_limits<int>::max();
    MPI_Allreduce(MPI_IN_PLACE, &first, 1, MPI_INT, MPI_MIN, comm);

    std::uint64_t length = rank == first ? m_message.size() : 0;
    MPI_Bcast(&length, 1, MPI_UINT64_T, first, comm);
    std::string message = rank == first ? m_message : std::string(length, ' ');
    MPI_Bcast(message.data(), static_cast<int>(length), MPI_CHAR, first, comm);

    std::ostringstream out;
    out << message << " (Found on rank " << first;
    if (failures > 1) {
      out << "; " << failures - 1 << (failures == 2 ? " more rank" : " more ranks")
          << " found a defect as well";
    }
    out << ".)";
    throw Error(out.str());
  }
#else  // USE_MPI
  /**
   * Raises the defect if this rank has found one.
   */
  void raise() const {
    if (failed()) {
      throw Error(m_message);
    }
  }
#endif // USE_MPI

  private:
  std::string m_message;
};

} // namespace internal

} // namespace PUML

#endif // PUML_ERROR_H
