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

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace PUML {

/**
 * Reports that the mesh, or the way it is being used, does not hold up.
 *
 * A caller that has no way to recover can let it escape, which ends the program
 * the way an aborting error did. One that wants to report the problem itself,
 * or a test that wants to provoke it, can catch it.
 *
 * Under MPI, a defect that only one rank can see is raised on that rank alone.
 * A caller that continues has to bring the other ranks along, or abort.
 */
class Error : public std::runtime_error {
  public:
  explicit Error(const std::string& what) : std::runtime_error(what) {}
};

namespace internal {

/// Builds the message of an Error from the pieces handed to it.
inline void formatError(std::ostringstream& out) { static_cast<void>(out); }

template <typename T, typename... Rest>
void formatError(std::ostringstream& out, const T& first, const Rest&... rest) {
  out << ' ' << first;
  formatError(out, rest...);
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

} // namespace PUML

#endif // PUML_ERROR_H
