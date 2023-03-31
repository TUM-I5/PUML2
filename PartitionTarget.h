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

    void set_vertex_weights_uniform(std::size_t vertex_count) {
        m_vertex_count = vertex_count;
        m_vertex_weights.clear();
    }

    void set_vertex_weights(const std::vector<double>& vertex_weights) {
        m_vertex_weights = vertex_weights;
        m_vertex_count = vertex_weights.size();
    }

    void set_vertex_weights(std::size_t vertex_count, double* vertex_weights) {
        m_vertex_count = vertex_count;
        m_vertex_weights = std::vector<double>(vertex_weights, vertex_weights + vertex_count);
    }

    void set_imbalance(double imbalance) {
        m_imbalance = imbalance;
    }

    const std::vector<double>& vertex_weights() {
        return m_vertex_weights;
    }

    bool vertex_weight_uniform() {
        return m_vertex_weights.empty();
    }

    bool edge_weight_uniform() {
        // TODO: implement non-uniform
        return true;
    }

    double imbalance() {
        return m_imbalance;
    }

    std::size_t vertex_count() {
        return m_vertex_count;
    }

private:
    std::vector<double> m_vertex_weights;
    std::size_t m_vertex_count;
    double m_imbalance;
};

}
#endif
