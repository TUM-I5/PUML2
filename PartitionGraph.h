/**
 * @file
 *  This file is part of PUML
 *
 *  For conditions of distribution and use, please see the copyright
 *  notice in the file 'COPYING' at the root directory of this package
 *  and the copyright notice at https://github.com/TUM-I5/PUMGen
 *
 * @copyright 2019-2023 Technische Universitaet Muenchen
 * @author David Schneller <david.schneller@tum.de>
 */

#ifndef PUML_PARTITION_GRAPH_H
#define PUML_PARTITION_GRAPH_H

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include <algorithm>
#include <vector>
#include <cassert>
#include <type_traits>
#include "Topology.h"
#include "PUML.h"
#include "FaceIterator.h"
#include "Downward.h"

#include "utils/logger.h"

namespace PUML
{

template<TopoType Topo>
class PartitionGraph
{
public:
    PartitionGraph(const PUML<Topo>& puml) : m_puml(puml) {
        int commSize = 1;
#ifdef USE_MPI
        m_comm = m_puml.comm();
        MPI_Comm_size(m_comm, &commSize);
#endif
        m_processCount = commSize;

        const unsigned long cellfaces = internal::Topology<Topo>::cellfaces();
        const auto& faces = m_puml.faces();
        const auto& cells = m_puml.cells();
        unsigned long vertexCount = cells.size();

        std::vector<unsigned long> adjRawCount(vertexCount);
        std::vector<unsigned long> adjRaw(vertexCount * cellfaces);

        FaceIterator<Topo> iterator(m_puml);
        iterator.template forEach<unsigned long>(
            [&cells] (int fid, int cid) {
                return cells[cid].gid();
            },
            [&adjRawCount, &adjRaw] (int id, int lid, const unsigned long& gid) {
                int idx = cellfaces * lid + adjRawCount[lid]++;
                adjRaw[idx] = gid;
            }
        );

        m_adjDisp.resize(vertexCount+1);
        m_adjDisp[0] = 0;
        std::inclusive_scan(adjRawCount.begin(), adjRawCount.end(), m_adjDisp.begin() + 1);

        m_adj.resize(m_adjDisp[vertexCount]);
        for (unsigned long i = 0, j = 0; i < vertexCount; ++i) {
            for (unsigned long k = 0; k < adjRawCount[i]; ++k, ++j) {
                m_adj[j] = adjRaw[i * cellfaces + k];
            }
        }

        m_vertexDistribution.resize(commSize+1);
        m_edgeDistribution.resize(commSize+1);
        m_vertexDistribution[0] = 0;
        m_edgeDistribution[0] = 0;

#ifdef USE_MPI
        MPI_Allgather(&vertexCount, 1, MPI_UNSIGNED_LONG, (unsigned long*)m_vertexDistribution.data() + 1, 1, MPI_UNSIGNED_LONG, m_comm);
        MPI_Allgather(&m_adjDisp[vertexCount], 1, MPI_UNSIGNED_LONG, (unsigned long*)m_edgeDistribution.data() + 1, 1, MPI_UNSIGNED_LONG, m_comm);

        for (unsigned long i = 2; i <= m_processCount; ++i)
        {
            m_vertexDistribution[i] += m_vertexDistribution[i - 1];
            m_edgeDistribution[i] += m_edgeDistribution[i - 1];
        }
#else
        m_vertexDistribution[1] = vertexCount;
        m_edgeDistribution[1] = m_adjDisp[vertexCount];
#endif
    }

    // FaceHandlerFunc: void(int,int,const T&,const T&,int)
    template<typename T, typename FaceHandlerFunc,
        std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, const T&, int>, bool> = true>
    void forEachLocalEdges(const T* cellData,
                        FaceHandlerFunc faceHandler,
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        auto handler = [&cellData] (int fid, int id){return cellData[id];};
        forEachLocalEdges<T>(handler, faceHandler, mpit);
    }

    // FaceHandlerFunc: void(int,int,const T&,const T&,int)
    template<typename T, typename FaceHandlerFunc,
        std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, const T&, int>, bool> = true>
    void forEachLocalEdges(const std::vector<T>& cellData,
                        FaceHandlerFunc faceHandler,
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        auto handler = [&cellData] (int fid, int id){return cellData[id];};
        forEachLocalEdges<T>(handler, faceHandler, mpit);
    }

    // CellHandlerFunc: T(int,int)
    // FaceHandlerFunc: void(int,int,const T&,const T&,int)
    template<typename T, typename CellHandlerFunc, typename FaceHandlerFunc,
        std::enable_if_t<std::is_invocable_r_v<T, CellHandlerFunc, int, int>, bool> = true,
        std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, const T&, int>, bool> = true>
    void forEachLocalEdges(CellHandlerFunc cellHandler,
                        FaceHandlerFunc faceHandler,
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        auto realFaceHandler = [&faceHandler, &cellHandler](int fid,int lid,const T& a, int eid) {
            faceHandler(fid, lid, a, cellHandler(fid,lid), eid);
        };
        forEachLocalEdges<T>(cellHandler, realFaceHandler, mpit);
    }

    // ExternalCellHandlerFunc: T(int,int)
    // FaceHandlerFunc: void(int,int,const T&,int)
    template<typename T, typename ExternalCellHandlerFunc, typename FaceHandlerFunc,
        std::enable_if_t<std::is_invocable_r_v<T, ExternalCellHandlerFunc, int, int>, bool> = true,
        std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, int>, bool> = true>
    void forEachLocalEdges(ExternalCellHandlerFunc externalCellHandler,
                        FaceHandlerFunc faceHandler,
                        MPI_Datatype mpit = MPITypeInfer<T>::type()) {
        
        std::vector<unsigned long> adjRawCount(localVertexCount());
        const auto& adjDisp = m_adjDisp;
        auto realFaceHandler = [&adjDisp, &adjRawCount, &faceHandler](int fid,int lid,const T& a) {
            faceHandler(fid, lid, a, adjDisp[lid] + adjRawCount[lid]++);
        };
        FaceIterator<Topo> iterator(m_puml);
        iterator.template forEach<T>(externalCellHandler, realFaceHandler, [](int a, int b){}, mpit);
    }

    unsigned long localVertexCount() const {
        return m_adjDisp.size() - 1;
    }

    unsigned long localEdgeCount() const {
        return m_adj.size();
    }

    unsigned long globalVertexCount() const {
        return m_vertexDistribution[m_processCount];
    }

    unsigned long globalEdgeCount() const {
        return m_edgeDistribution[m_processCount];
    }

    template<typename OutputType>
    void geometricCoordinates(std::vector<OutputType>& coord) const {
        // basic idea: compute the barycenter of the cell (i.e. tetrahedron/hexahedron)
        coord.resize(3 * localVertexCount());
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
    void setVertexWeights(const std::vector<T>& vertexWeights, int vertexWeightCount) {
        setVertexWeights(vertexWeights.data(), vertexWeightCount);
    }

    template<typename T>
    void setVertexWeights(const T* vertexWeights, int vertexWeightCount) {
        if (vertexWeights == nullptr) return;
        m_vertexWeightCount = vertexWeightCount;
        m_vertexWeights.resize(localVertexCount() * vertexWeightCount);
        for (size_t i = 0; i < m_vertexWeights.size(); ++i) {
            m_vertexWeights[i] = vertexWeights[i];
        }
    }

    template<typename T>
    void setEdgeWeights(const std::vector<T>& edgeWeights) {
        setEdgeWeights(edgeWeights.data());
    }

    template<typename T>
    void setEdgeWeights(const T* edgeWeights) {
        if (edgeWeights == nullptr) return;
        m_edgeWeights.resize(m_adj.size());
        for (size_t i = 0; i < m_adj.size(); ++i) {
            m_edgeWeights[i] = edgeWeights[i];
        }
    }

    const std::vector<unsigned long>& adj() const {
        return m_adj;
    }

    const std::vector<unsigned long>& adjDisp() const {
        return m_adjDisp;
    }

    const std::vector<unsigned long>& vertexDistribution() const {
        return m_vertexDistribution;
    }

    const std::vector<unsigned long>& edgeDistribution() const {
        return m_edgeDistribution;
    }

    const std::vector<unsigned long>& vertexWeights() const {
        return m_vertexWeights;
    }

    const std::vector<unsigned long>& edgeWeights() const {
        return m_edgeWeights;
    }

    const MPI_Comm& comm() const {
        return m_comm;
    }

    const PUML<Topo>& puml() const {
        return m_puml;
    }

    unsigned long vertexWeightCount() const {
        return m_vertexWeightCount;
    }

    unsigned long processCount() const {
        return m_processCount;
    }

private:
    std::vector<unsigned long> m_adj;
    std::vector<unsigned long> m_adjDisp;
    std::vector<unsigned long> m_vertexWeights;
    std::vector<unsigned long> m_edgeWeights;
    std::vector<unsigned long> m_vertexDistribution;
    std::vector<unsigned long> m_edgeDistribution;
    unsigned long m_vertexWeightCount = 0;
    unsigned long m_processCount = 0;
#ifdef USE_MPI
    MPI_Comm m_comm;
#endif
    const PUML<Topo>& m_puml;
};

using TETPartitionGraph = PartitionGraph<TETRAHEDRON>;

}
#endif
