// SPDX-FileCopyrightText: 2023 Technical University of Munich
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
 * @author David Schneller <david.schneller@tum.de>
 */

#ifndef PUML_PARTITION_TARGET_H
#define PUML_PARTITION_TARGET_H

#include <cstddef>
#ifdef USE_MPI
#endif // USE_MPI

#include <vector>
#include <cassert>

namespace PUML {

class PartitionTarget {
  public:
  PartitionTarget() = default;

  void setVertexWeightsUniform(std::size_t vertexCount) {
    m_vertexCount = vertexCount;
    m_vertexWeights.clear();
  }

  void setVertexWeights(const std::vector<double>& vertexWeights) {
    m_vertexWeights = vertexWeights;
    m_vertexCount = vertexWeights.size();
  }

  void setVertexWeights(std::size_t vertexCount, double* vertexWeights) {
    m_vertexCount = vertexCount;
    m_vertexWeights = std::vector<double>(vertexWeights, vertexWeights + vertexCount);
  }

  void setImbalance(double imbalance) { m_imbalance = imbalance; }

  [[nodiscard]] auto vertexWeights() const -> const std::vector<double>& { return m_vertexWeights; }

  [[nodiscard]] auto vertexWeightsUniform() const -> bool { return m_vertexWeights.empty(); }

  [[nodiscard]] auto imbalance() const -> double { return m_imbalance; }

  [[nodiscard]] auto vertexCount() const -> std::size_t { return m_vertexCount; }

  private:
  std::vector<double> m_vertexWeights;
  std::size_t m_vertexCount{};
  double m_imbalance{};
};

} // namespace PUML
#endif
