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

#ifndef PUML_DOWNELEMENT_H
#define PUML_DOWNELEMENT_H

#include <algorithm>
#include <cstring>
#include <functional>

namespace PUML::internal {

/**
 * A hashable element defined by the global ids of
 * the down elements
 *
 * @tparam N The number of down elements
 */
template <unsigned int N>
struct DownElement {
  /** The global ids of the downward elements */
  unsigned long down[N]{};

  DownElement(const unsigned long down[N]) {
    memcpy(this->down, down, N * sizeof(unsigned long));
    std::sort(this->down, this->down + N);
  }

  auto operator==(const DownElement& other) const -> bool {
    return memcmp(down, other.down, N * sizeof(unsigned long)) == 0;
  }
};

template <unsigned int N>
struct DownElementHash {
  auto operator()(const DownElement<N>& element) const -> std::size_t {
    std::size_t h = std::hash<unsigned long>{}(element.down[0]);
    for (unsigned int i = 1; i < N; i++) {
      hashCombine(h, element.down[i]);
    }

    return h;
  }

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

#endif // PUML_DOWNELEMENT_H
