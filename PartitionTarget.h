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

/**
 * Describes what a partitioning should produce: how many partitions, how much
 * each of them should be given, and how far apart they may end up.
 *
 * The weights here belong to the partitions, not to the cells of the mesh; a
 * per-cell weight is set on the graph instead.
 */
class PartitionTarget {
  public:
  PartitionTarget() = default;

  /// Asks for count partitions of equal size.
  void setPartitionCount(std::size_t count) {
    m_partitionCount = count;
    m_partitionWeights.clear();
  }

  /// Asks for one partition per weight, sized in proportion to it.
  void setPartitionWeights(const std::vector<double>& weights) {
    m_partitionWeights = weights;
    m_partitionCount = weights.size();
  }

  void setPartitionWeights(std::size_t count, const double* weights) {
    m_partitionCount = count;
    m_partitionWeights = std::vector<double>(weights, weights + count);
  }

  /// How far a partition may deviate from its share, as a fraction.
  void setImbalance(double imbalance) { m_imbalance = imbalance; }

  [[nodiscard]] auto partitionWeights() const -> const std::vector<double>& {
    return m_partitionWeights;
  }

  [[nodiscard]] auto partitionWeightsUniform() const -> bool { return m_partitionWeights.empty(); }

  [[nodiscard]] auto partitionCount() const -> std::size_t { return m_partitionCount; }

  [[nodiscard]] auto imbalance() const -> double { return m_imbalance; }

  [[deprecated("use setPartitionCount()")]] void setVertexWeightsUniform(std::size_t count) {
    setPartitionCount(count);
  }

  [[deprecated("use setPartitionWeights()")]] void
      setVertexWeights(const std::vector<double>& weights) {
    setPartitionWeights(weights);
  }

  [[deprecated("use setPartitionWeights()")]] void setVertexWeights(std::size_t count,
                                                                    double* weights) {
    setPartitionWeights(count, weights);
  }

  [[deprecated("use partitionWeights()")]] [[nodiscard]] auto vertexWeights() const
      -> const std::vector<double>& {
    return partitionWeights();
  }

  [[deprecated("use partitionWeightsUniform()")]] [[nodiscard]] auto vertexWeightsUniform() const
      -> bool {
    return partitionWeightsUniform();
  }

  [[deprecated("use partitionCount()")]] [[nodiscard]] auto vertexCount() const -> std::size_t {
    return partitionCount();
  }

  private:
  std::vector<double> m_partitionWeights;
  std::size_t m_partitionCount{0};
  double m_imbalance{0.05};
};

} // namespace PUML
#endif
