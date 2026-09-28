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

#ifndef PUML_UPWARDBUILDER_H
#define PUML_UPWARDBUILDER_H

#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

#include "Types.h"

namespace PUML::internal {

/**
 * Collects which entities sit above which, while the cells are walked.
 *
 * The relations are appended in whatever order they arise and are ordered once
 * at the end, so the whole relation costs two growing arrays instead of one
 * container per entity. The lists it hands out are sorted and free of
 * duplicates.
 */
class UpwardBuilder {
  public:
  /// Announces how many relations are going to be added.
  void reserve(Size relations) { m_pairs.reserve(relations); }

  void add(LocalId owner, LocalId value) { m_pairs.emplace_back(owner, value); }

  /**
   * Orders the relations of the given number of entities. Afterwards begin()
   * and end() describe the list of every entity.
   */
  void finish(Size owners) {
    // Count the relations of every entity and turn the counts into offsets.
    m_offsets.assign(owners + 2, 0);
    for (const auto& pair : m_pairs) {
      assert(pair.first < owners);
      ++m_offsets[pair.first + 2];
    }
    for (Size i = 2; i < m_offsets.size(); ++i) {
      m_offsets[i] += m_offsets[i - 1];
    }

    // Place every relation with its entity.
    m_values.resize(m_pairs.size());
    for (const auto& pair : m_pairs) {
      m_values[m_offsets[pair.first + 1]] = pair.second;
      ++m_offsets[pair.first + 1];
    }
    m_pairs.clear();
    m_pairs.shrink_to_fit();
    m_offsets.pop_back();

    // Order each list and drop the repetitions, compacting as we go. The write
    // position never overtakes the read position, because a list only shrinks.
    Size out = 0;
    for (Size owner = 0; owner < owners; ++owner) {
      const Size first = m_offsets[owner];
      const Size last = m_offsets[owner + 1];
      std::sort(m_values.begin() + first, m_values.begin() + last);

      const Size start = out;
      for (Size i = first; i < last; ++i) {
        if (out == start || m_values[out - 1] != m_values[i]) {
          m_values[out] = m_values[i];
          ++out;
        }
      }
      m_offsets[owner] = start;
    }
    m_offsets[owners] = out;
    m_values.resize(out);
  }

  [[nodiscard]] auto size(LocalId owner) const -> Size {
    return m_offsets[owner + 1] - m_offsets[owner];
  }

  [[nodiscard]] auto begin(LocalId owner) const -> const LocalId* {
    return m_values.data() + m_offsets[owner];
  }

  [[nodiscard]] auto end(LocalId owner) const -> const LocalId* {
    return m_values.data() + m_offsets[owner + 1];
  }

  void clear() {
    m_pairs.clear();
    m_pairs.shrink_to_fit();
    m_values.clear();
    m_values.shrink_to_fit();
    m_offsets.clear();
    m_offsets.shrink_to_fit();
  }

  private:
  std::vector<std::pair<LocalId, LocalId>> m_pairs;
  std::vector<LocalId> m_values;
  std::vector<Size> m_offsets;
};

} // namespace PUML::internal

#endif // PUML_UPWARDBUILDER_H
