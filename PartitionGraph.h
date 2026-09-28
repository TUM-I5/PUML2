// SPDX-FileCopyrightText: 2019-2023 Technical University of Munich
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

#ifndef PUML_PARTITION_GRAPH_H
#define PUML_PARTITION_GRAPH_H

#include <cstddef>
#include "TypeInference.h"
#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include <algorithm>
#include <vector>
#include <cassert>
#include <functional>
#include <numeric>
#include <type_traits>
#include <utility>
#include "Topology.h"
#include "PUML.h"
#include "FaceIterator.h"
#include "Downward.h"

namespace PUML {

template <TopoType Topo>
class PartitionGraph {
  public:
  PartitionGraph(const PUML<Topo>& puml) : m_puml(puml) {
    int commSize = 1;
#ifdef USE_MPI
    m_comm = m_puml.comm();
    MPI_Comm_size(m_comm, &commSize);
#endif
    m_processCount = commSize;

    const unsigned long cellfaces = internal::Topology<Topo>::cellfaces();
    const auto& cells = m_puml.cells();
    unsigned long vertexCount = cells.size();

    std::vector<unsigned long> adjRawCount(vertexCount);
    std::vector<unsigned long> adjRaw(vertexCount * cellfaces);

    FaceIterator<Topo> iterator(m_puml);
    iterator.template forEach<unsigned long>(
        [&cells](int /*fid*/, int cid) { return cells[cid].gid(); },
        [&adjRawCount, &adjRaw](int /*id*/, int lid, const unsigned long& gid) {
          const auto idx = (cellfaces * lid) + adjRawCount[lid]++;
          adjRaw[idx] = gid;
        });

    m_adjDisp.resize(vertexCount + 1);
    m_adjDisp[0] = 0;
    // Note: std::inclusive_scan can be used here but some compilers
    //  haven't provided support for it e.g., libc++@15.0.0
    m_adjDisp[1] = adjRawCount[0];
    for (std::size_t i = 1; i < adjRawCount.size(); ++i) {
      m_adjDisp[i + 1] = m_adjDisp[i] + adjRawCount[i];
    }

    // The face iterator visits the neighbours of a cell in the order of its local faces, and that
    // follows the order in which the mesh file lists the vertices of the cell. The partitioners
    // break ties in the order of the adjacency, so sort every row by the global id of the
    // neighbour: the partition then only depends on the cells, not on how their vertices are
    // listed. m_edgeOrder maps the position of an edge in iterator order to its sorted position.
    m_adj.resize(m_adjDisp[vertexCount]);
    m_edgeOrder.resize(m_adjDisp[vertexCount]);
    std::vector<unsigned long> row;
    for (unsigned long i = 0; i < vertexCount; ++i) {
      row.resize(adjRawCount[i]);
      std::iota(row.begin(), row.end(), 0UL);
      const auto* raw = &adjRaw[i * cellfaces];
      std::stable_sort(row.begin(), row.end(), [raw](auto a, auto b) { return raw[a] < raw[b]; });
      for (unsigned long k = 0; k < row.size(); ++k) {
        m_adj[m_adjDisp[i] + k] = raw[row[k]];
        m_edgeOrder[m_adjDisp[i] + row[k]] = m_adjDisp[i] + k;
      }
    }

    m_vertexDistribution.resize(commSize + 1);
    m_edgeDistribution.resize(commSize + 1);
    m_vertexDistribution[0] = 0;
    m_edgeDistribution[0] = 0;

#ifdef USE_MPI
    MPI_Allgather(&vertexCount,
                  1,
                  MPI_UNSIGNED_LONG,
                  static_cast<unsigned long*>(m_vertexDistribution.data()) + 1,
                  1,
                  MPI_UNSIGNED_LONG,
                  m_comm);
    MPI_Allgather(&m_adjDisp[vertexCount],
                  1,
                  MPI_UNSIGNED_LONG,
                  static_cast<unsigned long*>(m_edgeDistribution.data()) + 1,
                  1,
                  MPI_UNSIGNED_LONG,
                  m_comm);

    for (unsigned long i = 2; i <= m_processCount; ++i) {
      m_vertexDistribution[i] += m_vertexDistribution[i - 1];
      m_edgeDistribution[i] += m_edgeDistribution[i - 1];
    }
#else
    m_vertexDistribution[1] = vertexCount;
    m_edgeDistribution[1] = m_adjDisp[vertexCount];
#endif
  }

  // FaceHandlerFunc: void(int,int,const T&,const T&,int)
  template <
      typename T,
      typename FaceHandlerFunc,
      std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, const T&, int>,
                       bool> = true>
  void forEachLocalEdges(const T* cellData,
                         FaceHandlerFunc&& faceHandler,
                         MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    auto handler = [&cellData](int /*fid*/, int id) { return cellData[id]; };
    forEachLocalEdges<T>(std::move(handler), std::forward<FaceHandlerFunc>(faceHandler), mpit);
  }

  // FaceHandlerFunc: void(int,int,const T&,const T&,int)
  template <
      typename T,
      typename FaceHandlerFunc,
      std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, const T&, int>,
                       bool> = true>
  void forEachLocalEdges(const std::vector<T>& cellData,
                         FaceHandlerFunc&& faceHandler,
                         MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    auto handler = [&cellData](int /*fid*/, int id) { return cellData[id]; };
    forEachLocalEdges<T>(std::move(handler), std::forward<FaceHandlerFunc>(faceHandler), mpit);
  }

  // CellHandlerFunc: T(int,int)
  // FaceHandlerFunc: void(int,int,const T&,const T&,int)
  template <
      typename T,
      typename CellHandlerFunc,
      typename FaceHandlerFunc,
      std::enable_if_t<std::is_invocable_r_v<T, CellHandlerFunc, int, int>, bool> = true,
      std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, const T&, int>,
                       bool> = true>
  void forEachLocalEdges(CellHandlerFunc&& cellHandler,
                         FaceHandlerFunc&& faceHandler,
                         MPI_Datatype mpit = MPITypeInfer<T>::type()) {
    auto realFaceHandler = [faceHandler = std::forward<FaceHandlerFunc>(faceHandler),
                            cellHandler = std::forward<CellHandlerFunc>(cellHandler)](
                               int fid, int lid, const T& a, int eid) {
      auto b = std::invoke(cellHandler, fid, lid);
      std::invoke(faceHandler, fid, lid, a, b, eid);
    };
    forEachLocalEdges<T>(
        std::forward<CellHandlerFunc>(cellHandler), std::move(realFaceHandler), mpit);
  }

  // ExternalCellHandlerFunc: T(int,int)
  // FaceHandlerFunc: void(int,int,const T&,int)
  template <
      typename T,
      typename ExternalCellHandlerFunc,
      typename FaceHandlerFunc,
      std::enable_if_t<std::is_invocable_r_v<T, ExternalCellHandlerFunc, int, int>, bool> = true,
      std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, int>, bool> = true>
  void forEachLocalEdges(ExternalCellHandlerFunc&& externalCellHandler,
                         FaceHandlerFunc&& faceHandler,
                         MPI_Datatype mpit = MPITypeInfer<T>::type()) {

    std::vector<unsigned long> adjRawCount(localVertexCount());
    const auto& adjDisp = m_adjDisp;
    const auto& edgeOrder = m_edgeOrder;
    auto realFaceHandler =
        [&adjDisp,
         &adjRawCount,
         &edgeOrder,
         faceHandler = std::forward<FaceHandlerFunc>(faceHandler)](int fid, int lid, const T& a) {
          // the edge id is the position in the sorted adjacency (see the constructor)
          std::invoke(faceHandler, fid, lid, a, edgeOrder[adjDisp[lid] + adjRawCount[lid]++]);
        };
    FaceIterator<Topo> iterator(m_puml);
    iterator.template forEach<T>(std::forward<ExternalCellHandlerFunc>(externalCellHandler),
                                 std::move(realFaceHandler),
                                 std::move([](int /*a*/, int /*b*/) {}),
                                 mpit);
  }

  [[nodiscard]] auto localVertexCount() const -> unsigned long { return m_adjDisp.size() - 1; }

  [[nodiscard]] auto localEdgeCount() const -> unsigned long { return m_adj.size(); }

  [[nodiscard]] auto globalVertexCount() const -> unsigned long {
    return m_vertexDistribution[m_processCount];
  }

  [[nodiscard]] auto globalEdgeCount() const -> unsigned long {
    return m_edgeDistribution[m_processCount];
  }

  template <typename OutputType>
  void geometricCoordinates(std::vector<OutputType>& coord) const {
    // basic idea: compute the barycenter of the cell (i.e. tetrahedron/hexahedron); summed in
    // double over the vertices in the order of their global ids, so that it does not depend on
    // the order in which the mesh file lists the vertices of the cell
    constexpr auto CellVertices = internal::Topology<Topo>::cellvertices();
    const auto& vertices = m_puml.vertices();
    coord.resize(3 * localVertexCount());
    for (unsigned long i = 0; i < m_puml.cells().size(); ++i) {
      const auto& cell = m_puml.cells()[i];
      unsigned int lid[CellVertices];
      Downward::vertices(m_puml, cell, lid);
      // a cell of a mixed mesh which has fewer vertices than the widest kind leaves the rest
      // invalid; they are moved to the end and left out
      auto* const validEnd = std::remove(lid, lid + CellVertices, InvalidLocalId);
      std::sort(lid, validEnd, [&vertices](auto a, auto b) {
        return vertices[a].gid() < vertices[b].gid();
      });
      const auto count = static_cast<double>(validEnd - lid);
      double sum[3] = {0.0, 0.0, 0.0};
      for (const auto* vertex = lid; vertex != validEnd; ++vertex) {
        for (int d = 0; d < 3; ++d) {
          sum[d] += vertices[*vertex].coordinate()[d];
        }
      }
      for (int d = 0; d < 3; ++d) {
        coord[(i * 3) + d] = static_cast<OutputType>(sum[d] / count);
      }
    }
  }

  template <typename T>
  void setVertexWeights(const std::vector<T>& vertexWeights, int vertexWeightCount) {
    setVertexWeights(vertexWeights.data(), vertexWeightCount);
  }

  template <typename T>
  void setVertexWeights(const T* vertexWeights, int vertexWeightCount) {
    if (vertexWeights == nullptr) {
      return;
    }
    m_vertexWeightCount = vertexWeightCount;
    m_vertexWeights.resize(localVertexCount() * vertexWeightCount);
    for (size_t i = 0; i < m_vertexWeights.size(); ++i) {
      m_vertexWeights[i] = vertexWeights[i];
    }
  }

  template <typename T>
  void setEdgeWeights(const std::vector<T>& edgeWeights) {
    setEdgeWeights(edgeWeights.data());
  }

  template <typename T>
  void setEdgeWeights(const T* edgeWeights) {
    if (edgeWeights == nullptr) {
      return;
    }
    m_edgeWeights.resize(m_adj.size());
    for (size_t i = 0; i < m_adj.size(); ++i) {
      m_edgeWeights[i] = edgeWeights[i];
    }
  }

  [[nodiscard]] auto adj() const -> const std::vector<unsigned long>& { return m_adj; }

  [[nodiscard]] auto adjDisp() const -> const std::vector<unsigned long>& { return m_adjDisp; }

  [[nodiscard]] auto vertexDistribution() const -> const std::vector<unsigned long>& {
    return m_vertexDistribution;
  }

  [[nodiscard]] auto edgeDistribution() const -> const std::vector<unsigned long>& {
    return m_edgeDistribution;
  }

  [[nodiscard]] auto vertexWeights() const -> const std::vector<unsigned long>& {
    return m_vertexWeights;
  }

  [[nodiscard]] auto edgeWeights() const -> const std::vector<unsigned long>& {
    return m_edgeWeights;
  }

  [[nodiscard]] auto comm() const -> const MPI_Comm& { return m_comm; }

  auto puml() const -> const PUML<Topo>& { return m_puml; }

  [[nodiscard]] auto vertexWeightCount() const -> unsigned long { return m_vertexWeightCount; }

  [[nodiscard]] auto processCount() const -> unsigned long { return m_processCount; }

  private:
  std::vector<unsigned long> m_adj;
  std::vector<unsigned long> m_adjDisp;
  // position of an edge in face iterator order -> its position in m_adj
  std::vector<unsigned long> m_edgeOrder;
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

} // namespace PUML
#endif
