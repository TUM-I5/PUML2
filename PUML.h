// SPDX-FileCopyrightText: 2017-2024 Technical University of Munich
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

#ifndef PUML_PUML_H
#define PUML_PUML_H

#include "DataBuffer.h"
#include "TypeInference.h"
#include <cstddef>
#include <cstring>
#include <iterator>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <limits>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utils/logger.h"
#include "utils/stringutils.h"

#include "DownElement.h"
#include "Element.h"
#include "Numbering.h"
#include "Topology.h"
#include "VertexElementMap.h"

namespace PUML {

enum class DataType { Cell = 0, Vertex = 1 };

// some constexprs for legacy reasons
constexpr DataType CELL = DataType::Cell;
constexpr DataType VERTEX = DataType::Vertex;

/**
 * Describes how a number of mesh entities (i.e. elements or vertices) is spread
 * over the ranks.
 *
 * Two layouts are supported. An even split gives E/R+1 entities to the first
 * E%R of the R ranks and E/R entities to the remaining ones. An explicit split
 * is described by the entity offset of every rank, and lets a rank hold any
 * number of entities, including none.
 */
class Distributor {
  public:
  Distributor() = delete;

  /**
   * Creates an even split of newNumEntities entities over newNumRanks ranks.
   */
  Distributor(unsigned long newNumEntities, unsigned long newNumRanks)
      : numEntities(newNumEntities), numRanks(newNumRanks), entitiesPerRank(numEntities / numRanks),
        missingEntities(numEntities % numRanks) {
    assert(numRanks > 0);
  }

  /**
   * Creates an explicit split from the entity offset of every rank, followed by
   * the total number of entities. newOffsets therefore holds one entry more
   * than there are ranks, in ascending order.
   */
  explicit Distributor(std::vector<unsigned long> newOffsets)
      : numEntities(newOffsets.back()), numRanks(newOffsets.size() - 1), entitiesPerRank(0),
        missingEntities(0), offsets(std::move(newOffsets)) {
    assert(numRanks > 0);
    assert(std::is_sorted(offsets.begin(), offsets.end()));
  }

#ifdef USE_MPI
  /**
   * Creates an explicit split by gathering the entity count of every rank.
   */
  static auto fromLocalSize(unsigned long localSize, MPI_Comm comm) -> Distributor {
    int procs = 1;
    MPI_Comm_size(comm, &procs);

    std::vector<unsigned long> newOffsets(static_cast<std::size_t>(procs) + 1);
    newOffsets[0] = 0;
    MPI_Allgather(
        &localSize, 1, MPI_UNSIGNED_LONG, newOffsets.data() + 1, 1, MPI_UNSIGNED_LONG, comm);
    for (std::size_t i = 1; i < newOffsets.size(); ++i) {
      newOffsets[i] += newOffsets[i - 1];
    }
    return Distributor(std::move(newOffsets));
  }
#endif // USE_MPI

  /**
   * Gives the offset and size of data where the rank should read data.
   */
  [[nodiscard]] auto offsetAndSize(unsigned long rank) const
      -> std::pair<unsigned long, unsigned long> {
    assert(rank < numRanks);
    if (!offsets.empty()) {
      return {offsets[rank], offsets[rank + 1] - offsets[rank]};
    }

    unsigned long offset = 0;
    unsigned long size = 0;
    if (rank < missingEntities) {
      offset = rank * (entitiesPerRank + 1);
      size = std::min(entitiesPerRank + 1, numEntities - offset);
    } else {
      offset = missingEntities * (entitiesPerRank + 1) + (rank - missingEntities) * entitiesPerRank;
      size = std::min(entitiesPerRank, numEntities - offset);
    }
    assert(offset + size <= numEntities);
    return {offset, size};
  }

  /**
   * Gives the rank, which has read the entity with the given globalId.
   */
  [[nodiscard]] auto rankOfEntity(unsigned long globalId) const -> unsigned long {
    assert(globalId < numEntities);
    if (!offsets.empty()) {
      const auto it = std::upper_bound(offsets.begin(), offsets.end(), globalId);
      assert(it != offsets.begin());
      return static_cast<unsigned long>(std::distance(offsets.begin(), it)) - 1;
    }

    unsigned long rank = 0;
    if (globalId < missingEntities * (entitiesPerRank + 1)) {
      rank = globalId / (entitiesPerRank + 1);
    } else {
      rank =
          (globalId - missingEntities * (entitiesPerRank + 1)) / entitiesPerRank + missingEntities;
    }
    assert(rank < numRanks);
    return rank;
  }

  /**
   * Gives the local id of a mesh entity for the given globalId.
   */
  [[nodiscard]] auto globalToLocalId(unsigned long rank, unsigned long globalId) const
      -> unsigned long {
    assert(globalId < numEntities);
    auto [offset, size] = offsetAndSize(rank);
    assert(globalId >= offset);
    assert(globalId < offset + size);
    return globalId - offset;
  }

  /**
   * Gives the total number of entities over all ranks.
   */
  [[nodiscard]] auto totalSize() const -> unsigned long { return numEntities; }

  private:
  unsigned long numEntities;
  unsigned long numRanks;
  unsigned long entitiesPerRank;
  unsigned long missingEntities;

  /** The entity offsets of an explicit split; empty for an even split */
  std::vector<unsigned long> offsets;
};

/**
 * @todo Handle non-MPI case correct
 */
template <TopoType Topo>
class PUML {
  public:
  /** The cell type from the file */
  using ocell_t = std::array<unsigned long, internal::Topology<Topo>::cellvertices()>;

  /** The vertex type from the file */
  using overtex_t = std::array<double, internal::Topology<Topo>::dimension()>;

  /** Internal cell type */
  using cell_t = Cell<Topo>;

  /** Internal face type */
  using face_t = Face;

  /** Internal edge type */
  using edge_t = Edge;

  /** Internal vertex type */
  using vertex_t = Vertex<Topo>;

  private:
#ifdef USE_MPI
  MPI_Comm m_comm{MPI_COMM_WORLD};
#endif // USE_MPI

  /** The original number of cells/vertices on each node */
  std::array<unsigned int, 2> m_originalSize{};

  /** The original number of total cells/vertices */
  std::array<unsigned long, 2> m_originalTotalSize{};

  /** How the cells/vertices are spread over the ranks */
  std::array<std::optional<Distributor>, 2> m_distributor{};

  using g2l_t = std::unordered_map<unsigned long, unsigned int>;

  /** The list of all local cells */
  std::vector<cell_t> m_cells;

  /** The list of all local faces */
  std::vector<face_t> m_faces;

  /** Mapping from global face ids to local face ids */
  g2l_t m_facesg2l;

  /** List of all local edges */
  std::vector<edge_t> m_edges;

  /** Mapping from global edge ids to local edge ids */
  g2l_t m_edgesg2l;

  /** List of all local vertices */
  std::vector<vertex_t> m_vertices;

  /** Mapping from global vertex ids to locl vertex ids */
  g2l_t m_verticesg2l;

  /** Maps from local vertex ids to to a local face ids */
  internal::VertexElementMap<internal::Topology<Topo>::facevertices()> m_v2f;

  /** Maps from local vertex ids to local edge ids */
  internal::VertexElementMap<2> m_v2e;

  /** One vertex data array, as it was handed over and as it is spread out */
  struct VertexData {
    internal::DataBuffer original;
    internal::DataBuffer distributed;
  };

  /** User cell data */
  std::vector<internal::DataBuffer> m_cellData;

  /** User vertex data */
  std::vector<VertexData> m_vertexData;

  // data names; supersede number indexing
  std::unordered_map<std::string, std::size_t> m_cellDataIndex;

  // data names; supersede number indexing
  std::unordered_map<std::string, std::size_t> m_vertexDataIndex;

  int m_cellDataLegacyIndex{0};
  int m_vertexDataLegacyIndex{0};

  /**
   * Describes the split of the given number of local entities over all ranks.
   */
  [[nodiscard]] auto makeDistributor(std::size_t localSize) const -> Distributor {
#ifdef USE_MPI
    return Distributor::fromLocalSize(localSize, m_comm);
#else  // USE_MPI
    return Distributor(std::vector<unsigned long>{0, localSize});
#endif // USE_MPI
  }

  /**
   * Aborts if the number of entities of the given type is still unknown.
   */
  void requireEntityCount(DataType type) const {
    if (!m_distributor[static_cast<int>(type)].has_value()) {
      logError() << "The number of entities has to be known before data can be added; call "
                    "setSize or inferSize first.";
    }
  }

  /**
   * Files a data array under the given name, replacing an array of the same
   * name.
   */
  auto store(const std::string& name, DataType type, internal::DataBuffer&& buffer)
      -> internal::DataBuffer& {
    if (type == DataType::Vertex) {
      const auto existing = m_vertexDataIndex.find(name);
      if (existing != m_vertexDataIndex.end()) {
        m_vertexData[existing->second] = VertexData{std::move(buffer), {}};
        return m_vertexData[existing->second].original;
      }
      m_vertexDataIndex[name] = m_vertexData.size();
      m_vertexData.push_back(VertexData{std::move(buffer), {}});
      return m_vertexData.back().original;
    }

    const auto existing = m_cellDataIndex.find(name);
    if (existing != m_cellDataIndex.end()) {
      m_cellData[existing->second] = std::move(buffer);
      return m_cellData[existing->second];
    }
    m_cellDataIndex[name] = m_cellData.size();
    m_cellData.push_back(std::move(buffer));
    return m_cellData.back();
  }

  public:
  PUML() = default;
  ~PUML() = default;

  PUML(const PUML&) = delete;
  auto operator=(const PUML&) -> PUML& = delete;

  PUML(PUML&&) = default;
  auto operator=(PUML&&) -> PUML& = default;

  /**
   * Copies the mesh input: the entity counts, how they are spread over the
   * ranks, and every data array as it was handed over. The topology is not
   * copied, so generateMesh() has to run on the copy.
   *
   * @return A mesh that holds the same input without reading it again
   */
  [[nodiscard]] auto clone() const -> PUML {
    PUML copy;
#ifdef USE_MPI
    copy.m_comm = m_comm;
#endif // USE_MPI
    copy.m_originalSize = m_originalSize;
    copy.m_originalTotalSize = m_originalTotalSize;
    copy.m_distributor = m_distributor;

    copy.m_cellData = m_cellData;
    copy.m_cellDataIndex = m_cellDataIndex;
    copy.m_cellDataLegacyIndex = m_cellDataLegacyIndex;

    copy.m_vertexData.reserve(m_vertexData.size());
    for (const auto& array : m_vertexData) {
      copy.m_vertexData.push_back(VertexData{array.original, {}});
    }
    copy.m_vertexDataIndex = m_vertexDataIndex;
    copy.m_vertexDataLegacyIndex = m_vertexDataLegacyIndex;

    return copy;
  }

  /**
   * Reserves the values of a data array and gives write access to them, so that
   * the caller can fill them in place. An array of the same name is replaced.
   *
   * @param name The name to file the array under
   * @param type Whether the array covers cells or vertices
   * @param sizes The shape of the values of a single entity
   * @return The values of numOriginalCells() resp. numOriginalVertices()
   *         entities, each holding the product of sizes values
   */
  template <typename T>
  auto allocateData(const std::string& name,
                    DataType type,
                    const std::vector<size_t>& sizes
#ifdef USE_MPI
                    ,
                    MPI_Datatype mpiType = MPITypeInfer<T>::type()
#endif
                        ) -> T* {
    static_assert(std::is_trivially_copyable_v<T>, "T needs to be trivially copyable");
    static_assert(std::is_trivially_default_constructible_v<T>,
                  "T needs to be trivially default constructible");
    requireEntityCount(type);

    size_t elemSize = 1;
    for (auto size : sizes) {
      elemSize *= size;
    }

    internal::DataBuffer buffer(m_originalSize[static_cast<int>(type)],
                                elemSize,
                                sizeof(T)
#ifdef USE_MPI
                                    ,
                                mpiType
#endif // USE_MPI
    );
    return reinterpret_cast<T*>(store(name, type, std::move(buffer)).data());
  }

#ifdef USE_MPI
  void setComm(MPI_Comm comm) { m_comm = comm; }
#endif // USE_MPI

  /**
   * Gives the entity distribution for cells or vertices.
   */
  [[nodiscard]] auto distributor(DataType type) const -> const Distributor& {
    requireEntityCount(type);
    return *m_distributor[static_cast<int>(type)];
  }

  /**
   * Spreads the given total number of entities evenly over the ranks, which is
   * the split a reader of one shared file produces.
   */
  void setTotalSize(DataType type, std::size_t total) {
    int rank = 0;
    int procs = 1;
#ifdef USE_MPI
    MPI_Comm_rank(m_comm, &rank);
    MPI_Comm_size(m_comm, &procs);
#endif // USE_MPI

    const auto index = static_cast<int>(type);
    m_distributor[index] = Distributor(total, procs);
    m_originalTotalSize[index] = total;
    m_originalSize[index] = m_distributor[index]->offsetAndSize(rank).second;
  }

  /**
   * Sets the number of entities this rank holds.
   */
  void setSize(DataType type, std::size_t value) {
    const auto index = type == DataType::Vertex ? 1 : 0;
    m_originalSize[index] = value;
    m_distributor[index] = makeDistributor(value);
    m_originalTotalSize[index] = m_distributor[index]->totalSize();
  }

  /**
   * Reserves the next name of the legacy numbering for the given entity type.
   *
   * @return The name and the index it stands for
   */
  auto nextLegacyName(DataType type) -> std::pair<std::string, int> {
    int& counter = (type == DataType::Vertex) ? m_vertexDataLegacyIndex : m_cellDataLegacyIndex;
    const int index = counter;
    ++counter;
    return {"_" + std::to_string(index), index};
  }

  template <typename T>
  auto addDataArray(const T* rawData,
                    DataType type,
                    const std::vector<size_t>& sizes
#ifdef USE_MPI
                    ,
                    MPI_Datatype mpiType = MPITypeInfer<T>::type()
#endif
                        ) -> int {
    const auto [name, ret] = nextLegacyName(type);

    addDataArray<T>(name,
                    rawData,
                    type,
                    sizes
#ifdef USE_MPI
                    ,
                    mpiType
#endif
    );

    return ret;
  }

  template <typename T>
  void addDataArray(const std::string& name,
                    const T* rawData,
                    DataType type,
                    const std::vector<size_t>& sizes
#ifdef USE_MPI
                    ,
                    MPI_Datatype mpiType = MPITypeInfer<T>::type()
#endif
  ) {
    static_assert(std::is_trivially_copyable_v<T>, "T needs to be trivially copyable");
    static_assert(std::is_trivially_default_constructible_v<T>,
                  "T needs to be trivially default constructible");

    size_t elemSize = 1;
    for (auto size : sizes) {
      elemSize *= size;
    }

    // rawData is owned by the caller and holds exactly the entities this rank
    // has been assigned, which need not be an even share of the total.
    T* data = allocateData<T>(name,
                              type,
                              sizes
#ifdef USE_MPI
                              ,
                              mpiType
#endif // USE_MPI
    );
    std::memcpy(data, rawData, sizeof(T) * m_originalSize[static_cast<int>(type)] * elemSize);
  }

  void partition(const int* partition) {
    int rank = 0;
    int procs = 1;
#ifdef USE_MPI
    MPI_Comm_rank(m_comm, &rank);
    MPI_Comm_size(m_comm, &procs);
#endif // USE_MPI

    {
      // Create sorting indices
      std::vector<unsigned int> indices(m_originalSize[0]);
      std::iota(indices.begin(), indices.end(), 0);

      std::sort(indices.begin(), indices.end(), [&](const auto& i1, const auto& i2) {
        return partition[i1] < partition[i2];
      });

      // Sort cell data
      for (auto& array : m_cellData) {
        auto sorted = array.sameLayout(m_originalSize[0]);
        for (std::size_t i = 0; i < m_originalSize[0]; i++) {
          sorted.copyEntity(i, array, indices[i]);
        }
        array = std::move(sorted);
      }
    }

    // Compute exchange info
    std::vector<int> sendCount(procs);
    std::vector<int> recvCount(procs);

    for (std::size_t i = 0; i < m_originalSize[0]; i++) {
      assert(partition[i] < procs);
      ++sendCount[partition[i]];
    }

#ifdef USE_MPI
    MPI_Alltoall(sendCount.data(), 1, MPI_INT, recvCount.data(), 1, MPI_INT, m_comm);
#else  // USE_MPI
    recvCount[0] = sendCount[0];
#endif // USE_MPI

    std::vector<int> sDispls(procs);
    std::vector<int> rDispls(procs);
    sDispls[0] = 0;
    rDispls[0] = 0;
    for (int i = 1; i < procs; i++) {
      sDispls[i] = sDispls[i - 1] + sendCount[i - 1];
      rDispls[i] = rDispls[i - 1] + recvCount[i - 1];
    }

    m_originalSize[0] = rDispls[procs - 1] + recvCount[procs - 1];
    m_distributor[0] = makeDistributor(m_originalSize[0]);

#ifdef USE_MPI
    // Exchange cell data
    for (auto& array : m_cellData) {
      auto received = array.sameLayout(m_originalSize[0]);
      MPI_Alltoallv(array.data(),
                    sendCount.data(),
                    sDispls.data(),
                    array.mpiType(),
                    received.data(),
                    recvCount.data(),
                    rDispls.data(),
                    array.mpiType(),
                    m_comm);
      array = std::move(received);
    }
#endif // USE_MPI
  }

  void generateMesh() {
    distributeVertices({"connectivity"});
    constructGeometry("geometry");
    constructMesh("connectivity");
  }

  /**
    Distribute all vertex data (including geometric positions)
    to all cells that need it.
   */
  void distributeVertices(const std::vector<std::string>& indexDataNames) {
    int rank = 0;
    int procs = 1;
#ifdef USE_MPI
    MPI_Comm_rank(m_comm, &rank);
    MPI_Comm_size(m_comm, &procs);
#endif // USE_MPI

    const auto& vertexDistributor = distributor(DataType::Vertex);
    // Generate a list of vertices we need from other processors
    using IndexType = unsigned long;
    std::vector<std::unordered_set<unsigned long>> requiredVertexSets(procs);
    for (const auto& indexDataName : indexDataNames) {
      const auto& indexArray = m_cellData[m_cellDataIndex.at(indexDataName)];
      const auto elemCount = indexArray.entitySize() / sizeof(IndexType);
      const auto* data = reinterpret_cast<const IndexType*>(indexArray.data());
      for (std::size_t i = 0; i < m_originalSize[0]; i++) {
        for (std::size_t j = 0; j < elemCount; j++) {
          const auto index = data[i * elemCount + j];
          const auto proc = vertexDistributor.rankOfEntity(index);
          assert(proc < procs);

          requiredVertexSets[proc].insert(index);
        }
      }
    }

    // Generate information for requesting vertices
    std::size_t totalVertices = requiredVertexSets[0].size();
    for (int i = 1; i < procs; i++) {
      totalVertices += requiredVertexSets[i].size();
    }

    std::vector<int> sendCount(procs);
    std::vector<int> recvCount(procs);

    std::vector<unsigned long> requiredVertices(totalVertices);

    {
      std::size_t k = 0;
      for (int i = 0; i < procs; i++) {
        sendCount[i] = requiredVertexSets[i].size();

        for (const auto& it : requiredVertexSets[i]) {
          assert(k < totalVertices);
          requiredVertices[k] = it;
          ++k;
        }
      }
    }

    // Exchange required vertex information
#ifdef USE_MPI
    MPI_Alltoall(sendCount.data(), 1, MPI_INT, recvCount.data(), 1, MPI_INT, m_comm);
#else  // USE_MPI
    recvCount[0] = sendCount[0];
#endif // USE_MPI

    std::vector<int> sDispls(procs);
    std::vector<int> rDispls(procs);
    sDispls[0] = 0;
    rDispls[0] = 0;
    for (int i = 1; i < procs; i++) {
      sDispls[i] = sDispls[i - 1] + sendCount[i - 1];
      rDispls[i] = rDispls[i - 1] + recvCount[i - 1];
    }

    const unsigned int totalRecv = rDispls[procs - 1] + recvCount[procs - 1];

    std::vector<unsigned long> distribVertexIds(totalRecv);
#ifdef USE_MPI
    MPI_Alltoallv(requiredVertices.data(),
                  sendCount.data(),
                  sDispls.data(),
                  MPI_UNSIGNED_LONG,
                  distribVertexIds.data(),
                  recvCount.data(),
                  rDispls.data(),
                  MPI_UNSIGNED_LONG,
                  m_comm);
#else  // USE_MPI
    exchangeLocally(
        requiredVertices.data(), distribVertexIds.data(), totalRecv * sizeof(unsigned long));
#endif // USE_MPI

    // Send back vertex coordinates (and other data)
    std::vector<internal::DataBuffer> distribData;
    distribData.reserve(m_vertexData.size());
    for (const auto& array : m_vertexData) {
      distribData.push_back(array.original.sameLayout(totalRecv));
    }
    std::vector<std::vector<int>> sharedRanks(m_originalSize[1]);
    {
      std::size_t k = 0;
      for (int i = 0; i < procs; i++) {
        for (int j = 0; j < recvCount[i]; j++) {
          assert(k < totalRecv);
          distribVertexIds[k] = vertexDistributor.globalToLocalId(rank, distribVertexIds[k]);

          assert(distribVertexIds[k] < m_originalSize[1]);

          // Handle other vertex data
          for (std::size_t l = 0; l < m_vertexData.size(); l++) {
            distribData[l].copyEntity(k, m_vertexData[l].original, distribVertexIds[k]);
          }

          // Save all ranks for each vertex
          sharedRanks[distribVertexIds[k]].push_back(i);

          ++k;
        }
      }
    }

    for (auto& array : m_vertexData) {
      array.distributed = array.original.sameLayout(totalVertices);
    }
#ifdef USE_MPI
    for (std::size_t i = 0; i < m_vertexData.size(); i++) {
      MPI_Alltoallv(distribData[i].data(),
                    recvCount.data(),
                    rDispls.data(),
                    distribData[i].mpiType(),
                    m_vertexData[i].distributed.data(),
                    sendCount.data(),
                    sDispls.data(),
                    distribData[i].mpiType(),
                    m_comm);
    }
#else  // USE_MPI
    for (std::size_t i = 0; i < m_vertexData.size(); i++) {
      exchangeLocally(distribData[i].data(),
                      m_vertexData[i].distributed.data(),
                      m_vertexData[i].distributed.bytes());
    }
#endif // USE_MPI

    distribData.clear();

    // Send back the number of shared ranks for each vertex
    std::vector<unsigned int> distNsharedRanks(totalRecv);
    std::size_t distTotalSharedRanks = 0;
    for (unsigned int i = 0; i < totalRecv; i++) {
      assert(distribVertexIds[i] < m_originalSize[1]);
      distNsharedRanks[i] = sharedRanks[distribVertexIds[i]].size();
      distTotalSharedRanks += distNsharedRanks[i];
    }

    std::vector<unsigned int> recvNsharedRanks(totalVertices);
#ifdef USE_MPI
    MPI_Alltoallv(distNsharedRanks.data(),
                  recvCount.data(),
                  rDispls.data(),
                  MPI_UNSIGNED,
                  recvNsharedRanks.data(),
                  sendCount.data(),
                  sDispls.data(),
                  MPI_UNSIGNED,
                  m_comm);
#else  // USE_MPI
    exchangeLocally(
        distNsharedRanks.data(), recvNsharedRanks.data(), totalVertices * sizeof(unsigned int));
#endif // USE_MPI

    // Setup buffers for exchanging shared ranks
    std::vector<int> sharedSendCount(procs);
    std::vector<int> sharedRecvCount(procs);

    std::vector<int> distSharedRanks(distTotalSharedRanks);
    {
      std::size_t k = 0;
      std::size_t l = 0;
      for (int i = 0; i < procs; i++) {
        for (int j = 0; j < recvCount[i]; j++) {
          assert(k < totalRecv);
          assert(l + sharedRanks[distribVertexIds[k]].size() <= distTotalSharedRanks);
          memcpy(&distSharedRanks[l],
                 sharedRanks[distribVertexIds[k]].data(),
                 sharedRanks[distribVertexIds[k]].size() * sizeof(int));
          l += sharedRanks[distribVertexIds[k]].size();

          sharedSendCount[i] += sharedRanks[distribVertexIds[k]].size();

          ++k;
        }
      }
    }

    std::size_t recvTotalSharedRanks = 0;
    {
      std::size_t k = 0;
      for (int i = 0; i < procs; i++) {
        for (int j = 0; j < sendCount[i]; j++) {
          assert(k < totalVertices);
          recvTotalSharedRanks += recvNsharedRanks[k];
          sharedRecvCount[i] += recvNsharedRanks[k];

          ++k;
        }
      }
    }

    std::vector<int> recvSharedRanks(recvTotalSharedRanks);

    sDispls[0] = 0;
    rDispls[0] = 0;
    for (int i = 1; i < procs; i++) {
      sDispls[i] = sDispls[i - 1] + sharedSendCount[i - 1];
      rDispls[i] = rDispls[i - 1] + sharedRecvCount[i - 1];
    }

#ifdef USE_MPI
    MPI_Alltoallv(distSharedRanks.data(),
                  sharedSendCount.data(),
                  sDispls.data(),
                  MPI_INT,
                  recvSharedRanks.data(),
                  sharedRecvCount.data(),
                  rDispls.data(),
                  MPI_INT,
                  m_comm);
#else  // USE_MPI
    exchangeLocally(
        distSharedRanks.data(), recvSharedRanks.data(), recvTotalSharedRanks * sizeof(int));
#endif // USE_MPI

    // Generate the vertex array
    m_vertices.resize(totalVertices);

    {
      std::size_t k = 0;
      for (std::size_t i = 0; i < totalVertices; i++) {
        m_vertices[i].m_gid = requiredVertices[i];
        m_vertices[i].m_sharedRanks.resize(recvNsharedRanks[i] - 1);
        std::size_t l = 0;
        for (unsigned int j = 0; j < recvNsharedRanks[i]; j++) {
          if (recvSharedRanks[k] != rank) {
            m_vertices[i].m_sharedRanks[l] = recvSharedRanks[k];
            ++l;
          }
          ++k;
        }
        std::sort(m_vertices[i].m_sharedRanks.begin(), m_vertices[i].m_sharedRanks.end());
      }
    }

    // Construct to g2l map for the vertices
    constructG2L(m_vertices, m_verticesg2l);
  }

  /**
    Given all locally-needed vertex data, set up all geometric information.
    Not needed for purely-topological mesh construction.

    Deprecated.
   */
  void constructGeometry(const std::string& geometryName) {
    const auto* data = reinterpret_cast<const overtex_t*>(vertexData(geometryName));
    for (std::size_t i = 0; i < m_vertices.size(); ++i) {
      std::copy(data[i].begin(), data[i].end(), m_vertices[i].m_coordinate.begin());
    }
  }

  /**
    Given all locally-needed vertex data, construct edge and face topology.
   */
  void constructMesh(const std::string& cellDataName) {
    const auto* originalCells = reinterpret_cast<const ocell_t*>(cellData(cellDataName));

    // Create the cell, face and edge list
    m_cells.resize(m_originalSize[0]);
    m_v2f.clear();
    m_faces.clear();
    m_v2e.clear();

    unsigned long cellOffset = m_originalSize[0];
#ifdef USE_MPI
    MPI_Scan(MPI_IN_PLACE, &cellOffset, 1, MPI_UNSIGNED_LONG, MPI_SUM, m_comm);
#endif // USE_MPI
    cellOffset -= m_originalSize[0];

    {
      std::vector<std::set<unsigned int>> edgeUpward;
      std::vector<std::set<unsigned int>> vertexUpward(m_vertices.size());

      for (std::size_t i = 0; i < m_originalSize[0]; i++) {
        m_cells[i].m_gid = i + cellOffset;

        for (std::size_t j = 0; j < internal::Topology<Topo>::cellvertices(); j++) {
          m_cells[i].m_vertices[j] = m_verticesg2l[originalCells[i][j]];
        }

        // Faces
        std::array<unsigned int, internal::Topology<Topo>::dimension()> v{};
        std::array<unsigned int, internal::Topology<Topo>::cellfaces()> faces{};
        for (std::size_t j = 0; j < internal::Topology<Topo>::cellfaces(); ++j) {
          const auto& face = internal::Numbering<Topo>::facevertices()[j];
          for (std::size_t d = 0; d < internal::Topology<Topo>::dimension(); ++d) {
            v[d] = m_cells[i].m_vertices[face[d]];
          }
          faces[j] = addFace(m_v2f.add(v), i);
          if constexpr (internal::Topology<Topo>::dimension() == 2) {
            for (unsigned int d = 0; d < internal::Topology<Topo>::dimension(); ++d) {
              vertexUpward[v[d]].insert(faces[j]);
            }
          }
        }

        // Edges + Vertex upward information
        if constexpr (internal::Topology<Topo>::dimension() == 3) {
          std::array<unsigned int, internal::Topology<Topo>::dimension() - 1> w{};
          for (std::size_t j = 0; j < internal::Topology<Topo>::celledges(); ++j) {
            const auto& edge = internal::Numbering<Topo>::edgevertices()[j];
            const auto& edgeadj = internal::Numbering<Topo>::edgefaces()[j];
            w[0] = m_cells[i].m_vertices[edge[0]];
            w[1] = m_cells[i].m_vertices[edge[1]];
            const auto edgeIdx =
                addEdge(edgeUpward, m_v2e.add(w), faces[edgeadj[0]], faces[edgeadj[1]]);
            vertexUpward[w[0]].insert(edgeIdx);
            vertexUpward[w[1]].insert(edgeIdx);
          }
        }
      }

      // Create edges
      m_edges.clear();

      if constexpr (internal::Topology<Topo>::dimension() == 3) {
        m_edges.resize(edgeUpward.size());
        for (std::size_t i = 0; i < m_edges.size(); i++) {
          assert(m_edges[i].m_upward.empty());
          m_edges[i].m_upward.resize(edgeUpward[i].size());
          std::size_t j = 0;
          for (const auto& eu : edgeUpward[i]) {
            m_edges[i].m_upward[j] = eu;
            ++j;
          }
        }
        edgeUpward.clear(); // Free memory
      }

      // Set vertex upward information
      for (std::size_t i = 0; i < m_vertices.size(); i++) {
        m_vertices[i].m_upward.resize(vertexUpward[i].size());
        std::size_t j = 0;
        for (const auto& eu : vertexUpward[i]) {
          m_vertices[i].m_upward[j] = eu;
          ++j;
        }
      }
    }

    if constexpr (internal::Topology<Topo>::dimension() == 3) {
      // Generate shared information and global ids for edges
      generatedSharedAndGID<edge_t, vertex_t, 2>(m_edges, m_vertices);

      // Generate shared information and global ids for faces
      generatedSharedAndGID<face_t, edge_t, internal::Topology<Topo>::faceedges()>(m_faces,
                                                                                   m_edges);
    } else {
      // skip edges
      generatedSharedAndGID<face_t, vertex_t, 2>(m_faces, m_vertices);
    }
  }

  /**
   * @return The total number of all cells within the mesh.
   */
  auto numTotalCells() const -> unsigned int { return m_originalTotalSize[0]; }

  /**
   * @return The number of original cells on this rank
   *
   * @note This value can change when {@link partition()} is called
   */
  auto numOriginalCells() const -> unsigned int { return m_originalSize[0]; }

  /**
   * @return The number of original vertices on this rank
   */
  auto numOriginalVertices() const -> unsigned int { return m_originalSize[1]; }

  /**
   * @return The original cells on this rank
   *
   * @note The pointer gets invalid when {@link partition()} is called
   */
  auto originalCells() const -> const ocell_t* {
    return reinterpret_cast<const ocell_t*>(cellData("connectivity"));
  }

  /**
   * @return The original vertices on this rank
   */
  auto originalVertices() const -> const overtex_t* {
    return reinterpret_cast<const overtex_t*>(vertexData("geometry"));
  }

  /**
   * @return The cells of the mesh
   */
  auto cells() const -> const std::vector<cell_t>& { return m_cells; }

  /**
   * @return The facets/faces of the mesh (for 2D: edges)
   */
  auto faces() const -> const std::vector<face_t>& { return m_faces; }

  /**
   * @return The edges of the mesh (only relevant for 3D)
   */
  auto edges() const -> const std::vector<edge_t>& { return m_edges; }

  /**
   * @return The vertices of the mesh
   */
  auto vertices() const -> const std::vector<vertex_t>& { return m_vertices; }

  /**
   * @return User cell data
   */
  auto cellData(const std::string& name) const -> const void* {
    return m_cellData[m_cellDataIndex.at(name)].data();
  }

  /**
   * @return User vertex data
   */
  auto vertexData(const std::string& name) const -> const void* {
    return m_vertexData[m_vertexDataIndex.at(name)].distributed.data();
  }

  /**
   * @return User cell data
   */
  auto cellData(unsigned int index) const -> const void* {
    const std::string altName = "_" + std::to_string(index);
    return cellData(altName);
  }

  /**
   * @return User vertex data
   */
  auto vertexData(unsigned int index) const -> const void* {
    const std::string altName = "_" + std::to_string(index);
    return vertexData(altName);
  }

  /**
   * @param vertexIds A list of local vertex ids
   * @return The local face id for the given set of vertices or <code>-1</code> if
   *  the face does not exist
   */
  auto faceByVertices(
      const std::array<unsigned int, internal::Topology<Topo>::facevertices()>& vertexIds) const
      -> int {
    return m_v2f.find(vertexIds);
  }

#ifdef USE_MPI
  /**
   * @return The MPI communicator used by this class. (this field only exists, if MPI is enabled)
   */
  auto comm() const -> const MPI_Comm& { return m_comm; }
#endif

  private:
  /**
   * Add a face but only if it does not exist yet
   *
   * @param lid The local id of the face
   * @param plid The local id of the parent
   * @return The local id of the face
   */
  auto addFace(unsigned int lid, unsigned int plid) -> unsigned int {
    if (lid < m_faces.size()) {
      // Update an old face (but make sure that only happens once)

      if (m_faces[lid].m_upward[1] != -1) {
        logError() << "Mesh construction error: a face has more than two adjacent cells.";
      }

      m_faces[lid].m_upward[1] = plid;
      if (m_faces[lid].m_upward[1] < m_faces[lid].m_upward[0]) {
        std::swap(m_faces[lid].m_upward[0], m_faces[lid].m_upward[1]);
      }
    } else {
      // New face
      assert(lid == m_faces.size());

      face_t face;
      face.m_upward[0] = plid;
      face.m_upward[1] = -1;
      m_faces.push_back(face);
    }

    return lid;
  }

  /**
   * Generates the shared information and global ids from the downward elements
   *
   * @param elements The elements for which the data should be generated
   * @param down The downward elements
   * @tparam N The number of downward elements
   */
  template <typename E, typename D, std::size_t N>
  void generatedSharedAndGID(std::vector<E>& elements, const std::vector<D>& down) {
#ifdef USE_MPI
    std::vector<std::array<unsigned long, N>> downward(elements.size());

    MPI_Datatype type = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(N, MPI_UNSIGNED_LONG, &type);
    MPI_Type_commit(&type);

    {
      // Collect all shared ranks for each element and downward gids
      std::vector<std::array<const std::vector<int>*, N>> allShared(elements.size());

      {
        std::vector<std::size_t> downPos(elements.size());

        for (const auto& downElem : down) {
          for (const auto& upwardElem : downElem.m_upward) {
            assert(downPos[upwardElem] < N);
            allShared[upwardElem][downPos[upwardElem]] = &downElem.m_sharedRanks;
            downward[upwardElem][downPos[upwardElem]] = downElem.m_gid;
            ++downPos[upwardElem];
          }
        }
      }

      // Create the intersection of the shared ranks and update the elements
      assert(N >= 2);
      for (std::size_t i = 0; i < elements.size(); ++i) {
        assert(allShared[i][0] != nullptr);
        assert(allShared[i][1] != nullptr);

        std::set_intersection(allShared[i][0]->begin(),
                              allShared[i][0]->end(),
                              allShared[i][1]->begin(),
                              allShared[i][1]->end(),
                              std::back_inserter(elements[i].m_sharedRanks));

        std::vector<int> buffer;
        for (std::size_t j = 2; j < N; ++j) {
          buffer.clear();

          assert(allShared[i][j] != nullptr);
          std::set_intersection(elements[i].m_sharedRanks.begin(),
                                elements[i].m_sharedRanks.end(),
                                allShared[i][j]->begin(),
                                allShared[i][j]->end(),
                                std::back_inserter(buffer));

          std::swap(elements[i].m_sharedRanks, buffer);
        }
      }
    }

    for (auto& downData : downward) {
      internal::selectionSort<unsigned long, N>(downData.data());
    }

    // Eliminate false positves
    int rank = 0;
    int procs = 1;
    MPI_Comm_rank(m_comm, &rank);
    MPI_Comm_size(m_comm, &procs);

    std::vector<int> nShared(procs);
    for (const auto& element : elements) {
      for (const auto& rank : element.m_sharedRanks) {
        ++nShared[rank];
      }
    }

    std::vector<int> nRecvShared(procs);
    MPI_Alltoall(nShared.data(), 1, MPI_INT, nRecvShared.data(), 1, MPI_INT, m_comm);

    std::vector<int> sDispls(procs);
    std::vector<int> rDispls(procs);
    sDispls[0] = 0;
    rDispls[0] = 0;
    for (int i = 1; i < procs; i++) {
      sDispls[i] = sDispls[i - 1] + nShared[i - 1];
      rDispls[i] = rDispls[i - 1] + nRecvShared[i - 1];
    }

    const std::size_t totalShared = sDispls[procs - 1] + nShared[procs - 1];
    const std::size_t totalRecvShared = rDispls[procs - 1] + nRecvShared[procs - 1];

    {
      std::vector<std::array<unsigned long, N>> recvShared(totalRecvShared);

      {
        std::vector<std::array<unsigned long, N>> sendShared(totalShared);

        {
          std::vector<std::size_t> sharedPos(procs);

          for (std::size_t i = 0; i < elements.size(); ++i) {
            for (const auto& rank : elements[i].m_sharedRanks) {
              assert(sharedPos[rank] < static_cast<std::size_t>(nShared[rank]));
              sendShared[sDispls[rank] + sharedPos[rank]] = downward[i];
              ++sharedPos[rank];
            }
          }
        }

        MPI_Alltoallv(sendShared.data(),
                      nShared.data(),
                      sDispls.data(),
                      type,
                      recvShared.data(),
                      nRecvShared.data(),
                      rDispls.data(),
                      type,
                      m_comm);
      }

      {
        std::vector<std::unordered_set<internal::DownElement<N>, internal::DownElementHash<N>>>
            hashedElements(procs);

        unsigned int k = 0;
        for (int i = 0; i < procs; i++) {
          assert(i != rank || nRecvShared[i] == 0);
          for (int j = 0; j < nRecvShared[i]; j++) {
            assert(k < totalRecvShared);
            assert(hashedElements[i].find(recvShared[k]) == hashedElements[i].end());
            hashedElements[i].emplace(recvShared[k]);
            k++;
          }
        }

        for (std::size_t i = 0; i < elements.size(); i++) {
          auto it = elements[i].m_sharedRanks.begin();
          while (it != elements[i].m_sharedRanks.end()) {
            if (hashedElements[*it].find(downward[i]) == hashedElements[*it].end()) {
              it = elements[i].m_sharedRanks.erase(it);
            } else {
              ++it;
            }
          }
        }
      }
    }

    // Count owned elements
    std::size_t owned = 0;
    for (const auto& element : elements) {
      if (element.m_sharedRanks.empty() || element.m_sharedRanks[0] > rank) {
        ++owned;
      }
    }

    // Get global id offset
    unsigned long gidOffset = owned;
    MPI_Scan(MPI_IN_PLACE, &gidOffset, 1, MPI_UNSIGNED_LONG, MPI_SUM, m_comm);
    gidOffset -= owned;

    // Set global ids for owned elements and count the number of elements we need to forward
    std::vector<int> nSendGid(procs);
    std::vector<int> nRecvGid(procs);
    for (auto& element : elements) {
      if (element.m_sharedRanks.empty() || element.m_sharedRanks[0] > rank) {
        element.m_gid = gidOffset++;

        for (const auto& rank : element.m_sharedRanks) {
          nSendGid[rank]++;
        }
      } else {
        element.m_gid = std::numeric_limits<unsigned long>::max();

        if (!element.m_sharedRanks.empty()) {
          ++nRecvGid[element.m_sharedRanks[0]];
        }
      }
    }

    // Compute exchange offsets
    sDispls[0] = 0;
    rDispls[0] = 0;
    for (int i = 1; i < procs; i++) {
      sDispls[i] = sDispls[i - 1] + nSendGid[i - 1];
      rDispls[i] = rDispls[i - 1] + nRecvGid[i - 1];
    }

    const std::size_t totalSendGid = sDispls[procs - 1] + nSendGid[procs - 1];
    const std::size_t totalRecvGid = rDispls[procs - 1] + nRecvGid[procs - 1];
    std::unordered_map<internal::DownElement<N>, unsigned long, internal::DownElementHash<N>> dg2g;

    {
      std::vector<unsigned long> recvGid(totalRecvGid);
      std::vector<std::array<unsigned long, N>> recvDGid(totalRecvGid);

      // Collect send data
      {
        std::vector<unsigned long> sendGid(totalSendGid);
        std::vector<std::array<unsigned long, N>> sendDGid(totalSendGid);
        {
          std::vector<std::size_t> sendPos(procs);

          for (std::size_t i = 0; i < elements.size(); i++) {
            if (elements[i].m_sharedRanks.empty() || elements[i].m_sharedRanks[0] > rank) {
              for (const auto& rank : elements[i].m_sharedRanks) {
                assert(sendPos[rank] < static_cast<std::size_t>(nSendGid[rank]));

                sendGid[sDispls[rank] + sendPos[rank]] = elements[i].m_gid;
                sendDGid[sDispls[rank] + sendPos[rank]] = downward[i];
                ++sendPos[rank];
              }
            }
          }
        }

        MPI_Alltoallv(sendGid.data(),
                      nSendGid.data(),
                      sDispls.data(),
                      MPI_UNSIGNED_LONG,
                      recvGid.data(),
                      nRecvGid.data(),
                      rDispls.data(),
                      MPI_UNSIGNED_LONG,
                      m_comm);

        MPI_Alltoallv(sendDGid.data(),
                      nSendGid.data(),
                      sDispls.data(),
                      type,
                      recvDGid.data(),
                      nRecvGid.data(),
                      rDispls.data(),
                      type,
                      m_comm);
      }

      // Create a hash map from the received elements
      for (std::size_t i = 0; i < totalRecvGid; i++) {
        dg2g.emplace(recvDGid[i], recvGid[i]);
      }
    }

    // Assign gids
    for (std::size_t i = 0; i < elements.size(); i++) {
      if (!elements[i].m_sharedRanks.empty() && elements[i].m_sharedRanks[0] < rank) {
        assert(elements[i].m_gid == std::numeric_limits<unsigned long>::max());

        const auto it = dg2g.find(downward[i]);
        assert(it != dg2g.end());

        elements[i].m_gid = it->second;
      }
    }
    MPI_Type_free(&type);
#else  // USE_MPI
    static_cast<void>(down);

    // This rank owns every element, so the global ids are the local ones and no
    // element sits on a partition boundary.
    for (std::size_t i = 0; i < elements.size(); ++i) {
      elements[i].m_gid = i;
    }
#endif // USE_MPI
  }

  /**
   * Add an edge if it does not exist yet
   *
   * @param edgeUpward The storage for upward information
   * @param lid The local id of the edge
   * @param plid1 The first parent
   * @param plid2 The second parent
   * @return The local id of the edge
   */
  static auto addEdge(std::vector<std::set<unsigned int>>& edgeUpward,
                      unsigned int lid,
                      unsigned int plid1,
                      unsigned int plid2) -> unsigned int {
    if (lid >= edgeUpward.size()) {
      assert(lid == edgeUpward.size());
      edgeUpward.emplace_back();
    }
    edgeUpward[lid].insert(plid1);
    edgeUpward[lid].insert(plid2);

    return lid;
  }

  /**
   * Transfers an exchange buffer for the case that this rank is the only
   * participant in the exchange.
   *
   * @param send The send buffer
   * @param recv The receive buffer
   * @param bytes The number of bytes to transfer
   */
  static void exchangeLocally(const void* send, void* recv, std::size_t bytes) {
    if (bytes > 0) {
      std::memcpy(recv, send, bytes);
    }
  }

  /**
   * Constructs the global -> local map for an element array
   */
  template <typename TT>
  static void constructG2L(const std::vector<TT>& elements, g2l_t& g2lMap) {
    g2lMap.clear();

    unsigned int i = 0;
    for (typename std::vector<TT>::const_iterator it = elements.begin(); it != elements.end();
         ++it, i++) {
      assert(g2lMap.find(it->m_gid) == g2lMap.end());
      g2lMap[it->m_gid] = i;
    }
  }

  public:
  void identify(int dataId) {
    // use the convention for legacy names
    identify("connectivity", "_" + std::to_string(dataId));
  }

  /**
   * Writes a cell data array of vertex ids to a second array, with the ids of
   * the input mesh replaced by the local vertex ids of this rank.
   *
   * The vertices of the source array have to be part of the distribution, so
   * its name has to have been passed to distributeVertices().
   *
   * @param indexDataName The cell data array holding input vertex ids
   * @param localizedName The name to file the translated array under
   */
  void localize(const std::string& indexDataName, const std::string& localizedName) {
    using IndexType = unsigned long;

    const auto& source = m_cellData[m_cellDataIndex.at(indexDataName)];
    const auto elemCount = source.entitySize() / sizeof(IndexType);
    const auto* input = reinterpret_cast<const IndexType*>(source.data());

    auto* output = allocateData<IndexType>(localizedName, DataType::Cell, {elemCount});

    for (std::size_t i = 0; i < m_originalSize[0] * elemCount; ++i) {
      const auto local = m_verticesg2l.find(input[i]);
      if (local == m_verticesg2l.end()) {
        logError() << "Vertex" << input[i] << "of" << indexDataName
                   << "is not held by this rank; pass" << indexDataName << "to distributeVertices";
      }
      output[i] = local->second;
    }
  }

  void identify(const std::string& connectivityToUpdate, const std::string& identify) {
    const auto* identifiers = reinterpret_cast<const unsigned long*>(vertexData(identify));
    auto* connectivity =
        reinterpret_cast<ocell_t*>(m_cellData[m_cellDataIndex.at(connectivityToUpdate)].data());
    for (std::size_t i = 0; i < m_originalSize[0]; ++i) {
      for (std::size_t j = 0; j < internal::Topology<Topo>::cellvertices(); ++j) {
        connectivity[i][j] = identifiers[m_cells[i].m_vertices[j]];
      }
    }
  }
};

/** Convenient typedef for tetrahrdral meshes */
using TETPUML = PUML<TETRAHEDRON>;

} // namespace PUML

#endif // PUML_PUML_H
