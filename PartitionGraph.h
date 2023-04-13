/**
 * @file
 *  This file is part of PUML
 *
 *  For conditions of distribution and use, please see the copyright
 *  notice in the file 'COPYING' at the root directory of this package
 *  and the copyright notice at https://github.com/TUM-I5/PUMGen
 *
 * @copyright 2019 Technische Universitaet Muenchen
 * @author David Schneller <david.schneller@tum.de>
 */

#ifndef PUML_PARTITION_GRAPH_H
#define PUML_PARTITION_GRAPH_H

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include <vector>
#include <cassert>
#include "Topology.h"
#include "PUML.h"
#include "FaceIterator.h"

#include "utils/logger.h"

namespace PUML
{

template<TopoType Topo>
class PartitionGraph
{
public:
    PartitionGraph(const PUML<Topo>& puml) : m_puml(puml) {
        int comm_size = 1;
#ifdef USE_MPI
        m_comm = m_puml.comm();
        MPI_Comm_size(m_comm, &comm_size);
        m_process_count = comm_size;
#endif

        const unsigned long cellfaces = internal::Topology<Topo>::cellfaces();
        const auto& faces = m_puml.faces();
        const auto& cells = m_puml.cells();
        unsigned long vertex_count = cells.size();

        std::vector<unsigned long> adj_raw_count(vertex_count);
        std::vector<unsigned long> adj_raw(vertex_count * cellfaces);

        FaceIterator<Topo> iterator(m_puml);
        iterator.template iterate<unsigned long>(
            [&cells] (int fid, int cid) {
                return cells[cid].gid();
            },
            [&adj_raw_count, &adj_raw] (int id, int lid, const unsigned long& gid) {
                int idx = cellfaces * lid + adj_raw_count[lid]++;
                adj_raw[idx] = gid;
            }
        );

        m_adj_disp.resize(vertex_count+1);
        m_adj_disp[0] = 0;
        for (unsigned long i = 0; i < vertex_count; ++i) {
            m_adj_disp[i+1] = m_adj_disp[i] + adj_raw_count[i];
        }

        m_adj.resize(m_adj_disp[vertex_count]);
        for (unsigned long i = 0, j = 0; i < vertex_count; ++i) {
            for (unsigned long k = 0; k < adj_raw_count[i]; ++k, ++j) {
                m_adj[j] = adj_raw[i * cellfaces + k];
            }
        }

        m_vertex_distribution.resize(comm_size+1);
        m_edge_distribution.resize(comm_size+1);
        m_vertex_distribution[0] = 0;
        m_edge_distribution[0] = 0;

#ifdef USE_MPI
        MPI_Allgather(&vertex_count, 1, MPI_UNSIGNED_LONG, m_vertex_distribution.data() + 1, 1, MPI_UNSIGNED_LONG, m_comm);
        MPI_Allgather(&m_adj_disp[vertex_count], 1, MPI_UNSIGNED_LONG, m_edge_distribution.data() + 1, 1, MPI_UNSIGNED_LONG, m_comm);
#else
        m_vertex_distribution[1] = vertex_count;
        m_edge_distribution[1] = m_adj_disp[vertex_count];
#endif
        for (unsigned long i = 2; i <= m_process_count; ++i)
        {
            m_vertex_distribution[i] += m_vertex_distribution[i - 1];
            m_edge_distribution[i] += m_edge_distribution[i - 1];
        }
    }

    template<typename T>
    void forall_local_edges(const T* cell_data,
                        const std::function<void(int,int,const T&,const T&,int)>& face_handler,
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        const auto& handler = [&cell_data] (int fid, int id) -> const T& {return cell_data[id];};
        forall_local_edges<T>(handler, face_handler, mpit);
    }

    template<typename T>
    void forall_local_edges(const std::vector<T>& cell_data,
                        const std::function<void(int,int,const T&,const T&,int)>& face_handler,
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        const auto& handler = [&cell_data] (int fid, int id) -> const T& {return cell_data[id];};
        forall_local_edges<T>(handler, face_handler, mpit);
    }

    template<typename T>
    void forall_local_edges(const std::function<T(int,int)>& cell_handler,
                        const std::function<void(int,int,const T&,const T&,int)>& face_handler,
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        const auto& real_face_handler = [&face_handler, &cell_handler](int fid,int lid,const T& a, int eid) {
            face_handler(fid, lid, a, cell_handler(fid,lid), eid);
        };
        forall_local_edges<T>(cell_handler, real_face_handler, mpit);
    }

    template<typename T>
    void forall_local_edges(const std::function<T(int,int)>& external_cell_handler,
                        const std::function<void(int,int,const T&,int)>& face_handler,
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        
        std::vector<unsigned long> adj_raw_count(local_vertex_count());
        const auto& adj_disp = m_adj_disp;
        const auto& real_face_handler = [&adj_disp, &adj_raw_count, &face_handler](int fid,int lid,const T& a) {
            face_handler(fid, lid, a, adj_disp[lid] + adj_raw_count[lid]++);
        };
        FaceIterator<Topo> iterator(m_puml);
        iterator.template iterate<T>(external_cell_handler, real_face_handler, [](int a, int b){}, mpit);
    }

    unsigned long local_vertex_count() const {
        return m_adj_disp.size() - 1;
    }

    unsigned long local_edge_count() const {
        return m_adj.size();
    }

    unsigned long global_vertex_count() const {
        return m_vertex_distribution[m_process_count];
    }

    unsigned long global_edge_count() const {
        return m_edge_distribution[m_process_count];
    }

    template<typename OutputType>
    void geometric_coordinates(std::vector<OutputType>& coord) const {
        // basic idea: compute the barycenter of the cell (i.e. tetrahedron/hexahedron)
        coord.resize(3 * local_vertex_count());
        for (unsigned long i = 0; i < m_puml.cells().size(); ++i) {
            const auto& cell = m_puml.cells()[i];
            unsigned int lid[internal::Topology<Topo>::cellvertices()];
            Downward::vertices(m_puml, cell, lid);
            OutputType x = 0.0, y = 0.0, z = 0.0;
            for (unsigned long j = 0; j < internal::Topology<Topo>::cellvertices(); ++j) {
                auto vertex = m_puml.vertices()[lid[j]];
                x += vertex.coordinate()[0];
                y += vertex.coordinate()[1];
                z += vertex.coordinate()[2];
            }
            x /= internal::Topology<Topo>::cellvertices();
            y /= internal::Topology<Topo>::cellvertices();
            z /= internal::Topology<Topo>::cellvertices();
            coord[i * 3 + 0] = x;
            coord[i * 3 + 1] = y;
            coord[i * 3 + 2] = z;
        }
    }

    template<typename T>
    void set_vertex_weights(const std::vector<T>& vertex_weights, int vertex_weight_count) {
        set_vertex_weights(vertex_weights.data(), vertex_weight_count);
    }

    template<typename T>
    void set_vertex_weights(const T* vertex_weights, int vertex_weight_count) {
        if (vertex_weights == nullptr) return;
        m_vertex_weight_count = vertex_weight_count;
        m_vertex_weights.resize(local_vertex_count() * vertex_weight_count);
        for (size_t i = 0; i < m_vertex_weights.size(); ++i) {
            m_vertex_weights[i] = vertex_weights[i];
        }
    }

    template<typename T>
    void set_edge_weights(const std::vector<T>& edge_weights) {
        set_edge_weights(edge_weights.data());
    }

    template<typename T>
    void set_edge_weights(const T* edge_weights) {
        if (edge_weights == nullptr) return;
        m_edge_weights.resize(m_adj.size());
        for (size_t i = 0; i < m_adj.size(); ++i) {
            m_edge_weights[i] = edge_weights[i];
        }
    }

    const std::vector<unsigned long>& adj() const {
        return m_adj;
    }

    const std::vector<unsigned long>& adj_disp() const {
        return m_adj_disp;
    }

    const std::vector<unsigned long>& vertex_distribution() const {
        return m_vertex_distribution;
    }

    const std::vector<unsigned long>& edge_distribution() const {
        return m_edge_distribution;
    }

    const std::vector<unsigned long>& vertex_weights() const {
        return m_vertex_weights;
    }

    const std::vector<unsigned long>& edge_weights() const {
        return m_edge_weights;
    }

    const MPI_Comm& comm() const {
        return m_comm;
    }

    const PUML<Topo>& puml() const {
        return m_puml;
    }

    unsigned long vertex_weight_count() const {
        return m_vertex_weight_count;
    }

    unsigned long process_count() const {
        return m_process_count;
    }

private:
    std::vector<unsigned long> m_adj;
    std::vector<unsigned long> m_adj_disp;
    std::vector<unsigned long> m_vertex_weights;
    std::vector<unsigned long> m_edge_weights;
    std::vector<unsigned long> m_vertex_distribution;
    std::vector<unsigned long> m_edge_distribution;
    unsigned long m_vertex_weight_count = 0;
    unsigned long m_process_count = 0;
#ifdef USE_MPI
    MPI_Comm m_comm;
#endif
    const PUML<Topo>& m_puml;
};

using TETPartitionGraph = PartitionGraph<TETRAHEDRON>;

}
#endif
