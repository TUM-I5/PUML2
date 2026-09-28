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

#include "TypeInference.h"
#include <cstddef>
#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include "Downward.h"
#include "Error.h"
#include "FaceIterator.h"
#include "PUML.h"
#include "Topology.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

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

    // A rank without cells has no neighbour counts at all, only the leading 0.
    m_adjDisp.resize(vertexCount + 1);
    m_adjDisp[0] = 0;
    // Note: std::inclusive_scan can be used here but some compilers
    //  haven't provided support for it e.g., libc++@15.0.0
    for (std::size_t i = 0; i < adjRawCount.size(); ++i) {
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
                         FaceHandlerFunc&& faceHandler
#ifdef USE_MPI
                         ,
                         MPI_Datatype mpit = MPITypeInfer<T>::type()
#endif // USE_MPI
  ) {
    auto handler = [&cellData](int /*fid*/, int id) { return cellData[id]; };
    forEachLocalEdges<T>(std::move(handler),
                         std::forward<FaceHandlerFunc>(faceHandler)
#ifdef USE_MPI
                             ,
                         mpit
#endif // USE_MPI
    );
  }

  // FaceHandlerFunc: void(int,int,const T&,const T&,int)
  template <
      typename T,
      typename FaceHandlerFunc,
      std::enable_if_t<std::is_invocable_v<FaceHandlerFunc, int, int, const T&, const T&, int>,
                       bool> = true>
  void forEachLocalEdges(const std::vector<T>& cellData,
                         FaceHandlerFunc&& faceHandler
#ifdef USE_MPI
                         ,
                         MPI_Datatype mpit = MPITypeInfer<T>::type()
#endif // USE_MPI
  ) {
    auto handler = [&cellData](int /*fid*/, int id) { return cellData[id]; };
    forEachLocalEdges<T>(std::move(handler),
                         std::forward<FaceHandlerFunc>(faceHandler)
#ifdef USE_MPI
                             ,
                         mpit
#endif // USE_MPI
    );
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
                         FaceHandlerFunc&& faceHandler
#ifdef USE_MPI
                         ,
                         MPI_Datatype mpit = MPITypeInfer<T>::type()
#endif // USE_MPI
  ) {
    // The cell handler gives the values of both cells of an edge: of the one on the far side
    // through the face iterator, and of the one on the near side here. Neither may take it over
    // from the other, so both refer to the handler that was passed, which outlives this call.
    auto realFaceHandler = [faceHandler = std::forward<FaceHandlerFunc>(faceHandler),
                            &cellHandler](int fid, int lid, const T& a, int eid) {
      auto b = std::invoke(cellHandler, fid, lid);
      std::invoke(faceHandler, fid, lid, a, b, eid);
    };
    forEachLocalEdges<T>(cellHandler,
                         std::move(realFaceHandler)
#ifdef USE_MPI
                             ,
                         mpit
#endif // USE_MPI
    );
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
                         FaceHandlerFunc&& faceHandler
#ifdef USE_MPI
                         ,
                         MPI_Datatype mpit = MPITypeInfer<T>::type()
#endif // USE_MPI
  ) {

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
                                 std::move([](int /*a*/, int /*b*/) {})
#ifdef USE_MPI
                                     ,
                                 mpit
#endif // USE_MPI
    );
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
    const auto& vertices = m_puml.vertices();
    coord.resize(3 * localVertexCount());
    for (unsigned long i = 0; i < m_puml.cells().size(); ++i) {
      const auto& cell = m_puml.cells()[i];
      auto lid = Downward::vertices(m_puml, cell);
      // a cell of a mixed mesh which has fewer vertices than the widest kind leaves the rest
      // invalid; they are sorted to the end and left out. Sorting all of the slots, rather than
      // the valid ones only, keeps the length of the range known while compiling.
      std::sort(lid.begin(), lid.end(), [&vertices](auto a, auto b) {
        if (b == InvalidLocalId) {
          return a != InvalidLocalId;
        }
        if (a == InvalidLocalId) {
          return false;
        }
        return vertices[a].gid() < vertices[b].gid();
      });
      double count = 0.0;
      double sum[3] = {0.0, 0.0, 0.0};
      for (const auto vertex : lid) {
        if (vertex == InvalidLocalId) {
          break;
        }
        for (int d = 0; d < 3; ++d) {
          sum[d] += vertices[vertex].coordinate()[d];
        }
        count += 1.0;
      }
      for (int d = 0; d < 3; ++d) {
        coord[(i * 3) + d] = static_cast<OutputType>(sum[d] / count);
      }
    }
  }

  /**
   * Gives every local vertex vertexWeightCount weights, stored vertex by vertex.
   *
   * The partitioners have to be told the same number of weights on every rank, and a rank cannot
   * see what the others passed. So every rank has to call this with the same vertexWeightCount, a
   * rank without cells included: it is the count that says whether there are weights, and a rank
   * without cells takes it over although it has no weights to give. A count of 0 removes the
   * weights.
   */
  template <typename T>
  void setVertexWeights(const std::vector<T>& vertexWeights, int vertexWeightCount) {
    if (vertexWeightCount > 0 &&
        vertexWeights.size() < localVertexCount() * static_cast<unsigned long>(vertexWeightCount)) {
      throwError("the graph needs",
                 vertexWeightCount,
                 "weights for each of its",
                 localVertexCount(),
                 "local vertices, but got",
                 vertexWeights.size());
    }
    setVertexWeights(vertexWeights.data(), vertexWeightCount);
  }

  /**
   * As above, with the localVertexCount() * vertexWeightCount weights read from vertexWeights.
   *
   * The pointer is read from only if there is something to read, so a rank without cells may pass
   * nullptr, as data() of an empty vector may be. It does not stand for "no weights" there: the
   * count is taken over all the same.
   */
  template <typename T>
  void setVertexWeights(const T* vertexWeights, int vertexWeightCount) {
    if (vertexWeightCount < 0) {
      throwError("the number of weights per vertex cannot be negative, but got", vertexWeightCount);
    }
    const auto size = localVertexCount() * static_cast<unsigned long>(vertexWeightCount);
    if (vertexWeights == nullptr && size > 0) {
      throwError("the graph needs", size, "vertex weights, but got none");
    }
    m_vertexWeightCount = vertexWeightCount;
    m_vertexWeights.resize(size);
    for (size_t i = 0; i < m_vertexWeights.size(); ++i) {
      m_vertexWeights[i] = vertexWeights[i];
    }
  }

  /**
   * Gives every local edge a weight, in the order of adj().
   *
   * Like the vertex weights, the edge weights are a setting of the whole graph: every rank has to
   * call this, a rank without edges included.
   */
  template <typename T>
  void setEdgeWeights(const std::vector<T>& edgeWeights) {
    if (edgeWeights.size() < localEdgeCount()) {
      throwError("the graph needs a weight for each of its",
                 localEdgeCount(),
                 "local edges, but got",
                 edgeWeights.size());
    }
    setEdgeWeights(edgeWeights.data());
  }

  /**
   * As above, with the localEdgeCount() weights read from edgeWeights.
   *
   * As for the vertex weights, the pointer is read from only if there is something to read: a
   * rank without edges may pass nullptr, and has edge weights all the same.
   */
  template <typename T>
  void setEdgeWeights(const T* edgeWeights) {
    if (edgeWeights == nullptr && localEdgeCount() > 0) {
      throwError("the graph needs", localEdgeCount(), "edge weights, but got none");
    }
    m_hasEdgeWeights = true;
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

#ifdef USE_MPI
  [[nodiscard]] auto comm() const -> const MPI_Comm& { return m_comm; }
#endif // USE_MPI

  [[nodiscard]] auto puml() const -> const PUML<Topo>& { return m_puml; }

  [[nodiscard]] auto vertexWeightCount() const -> unsigned long { return m_vertexWeightCount; }

  /// Whether setEdgeWeights was called; the same on all ranks, those without edges included.
  [[nodiscard]] auto hasEdgeWeights() const -> bool { return m_hasEdgeWeights; }

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
  bool m_hasEdgeWeights = false;
  unsigned long m_processCount = 0;
#ifdef USE_MPI
  MPI_Comm m_comm;
#endif
  const PUML<Topo>& m_puml;
};

using TETPartitionGraph = PartitionGraph<TETRAHEDRON>;

} // namespace PUML
#endif
