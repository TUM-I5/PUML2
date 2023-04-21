/**
 * @file
 *  This file is part of PUML
 *
 *  For conditions of distribution and use, please see the copyright
 *  notice in the file 'COPYING' at the root directory of this package
 *  and the copyright notice at https://github.com/TUM-I5/PUMGen
 *
 * @copyright 2023 Technische Universitaet Muenchen
 * @author David Schneller <david.schneller@tum.de>
 */

#ifndef PUML_PARTITION_TARGET_H
#define PUML_PARTITION_TARGET_H

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include <vector>
#include <cassert>
#include "Topology.h"
#include "PUML.h"

#include "utils/logger.h"

namespace PUML
{

class PartitionTarget {
public:
    PartitionTarget() {}

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

    void setImbalance(double imbalance) {
        m_imbalance = imbalance;
    }

    const std::vector<double>& vertexWeights() const {
        return m_vertexWeights;
    }

    bool vertexWeightsUniform() const {
        return m_vertexWeights.empty();
    }

    bool edgeWeightsUniform() const {
        // TODO: implement non-uniform
        return true;
    }

    double imbalance() const {
        return m_imbalance;
    }

    std::size_t vertexCount() const {
        return m_vertexCount;
    }

private:
    std::vector<double> m_vertexWeights;
    std::size_t m_vertexCount;
    double m_imbalance;
};

}
#endif
