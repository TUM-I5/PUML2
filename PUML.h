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

#include "TypeInference.h"
#include <cstddef>
#include <cstring>
#include <hdf5.h>
#include <iterator>
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

enum DataType { CELL = 0, VERTEX = 1 };

/**
 * Distributes a number of mesh entities (i.e. elements or vertices) to  a given number of ranks
 * For E entities and R ranks, the first E%R ranks will read E/R+1 entities.
 * The remaining ranks will read E/R entities.
 */
class Distributor {
  public:
  Distributor() = delete;
  Distributor(unsigned long newNumEntities, unsigned long newNumRanks)
      : numEntities(newNumEntities), numRanks(newNumRanks), entitiesPerRank(numEntities / numRanks),
        missingEntities(numEntities % numRanks) {
    assert(numEntities > numRanks);
  }

  /**
   * Gives the offset and size of data where the rank should read data.
   */
  [[nodiscard]] auto offsetAndSize(unsigned long rank) const
      -> std::pair<unsigned long, unsigned long> {
    assert(rank < numRanks);
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

  private:
  unsigned long numEntities;
  unsigned long numRanks;
  unsigned long entitiesPerRank;
  unsigned long missingEntities;
};

#define checkH5Err(...) checkH5ErrImpl(__VA_ARGS__, __FILE__, __LINE__, rank)

/**
 * @todo Handle non-MPI case correct
 */
template <TopoType Topo>
class PUML {
  public:
  /** The cell type from the file */
  using ocell_t = unsigned long[internal::Topology<Topo>::cellvertices()];

  /** The vertex type from the file */
  using overtex_t = double[internal::Topology<Topo>::dimension()];

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

  /** The original cells from the file */
  ocell_t* m_originalCells{nullptr};

  /** The original vertices from the file */
  overtex_t* m_originalVertices{nullptr};

  /** The original number of cells/vertices on each node */
  unsigned int m_originalSize[2]{};

  /** The original number of total cells/vertices */
  unsigned long m_originalTotalSize[2]{};

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

  /** User cell data */
  std::vector<void*> m_cellData;

  /** User vertex data */
  std::vector<void*> m_vertexData;

  /** Original user vertex data */
  std::vector<void*> m_originalVertexData;

  std::vector<std::size_t> m_cellDataSize;

  std::vector<std::size_t> m_vertexDataSize;

#ifdef USE_MPI

  std::vector<MPI_Datatype> m_cellDataType;
  std::vector<bool> m_cellDataTypeDerived;

  std::vector<MPI_Datatype> m_vertexDataType;
  std::vector<bool> m_vertexDataTypeDerived;
#endif

  auto createDatatypeArray(MPI_Datatype baseType, std::size_t elemSize)
      -> std::pair<MPI_Datatype, bool> {
    if (elemSize == 1) {
      return {baseType, false};
    }
    MPI_Datatype newType = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(elemSize, baseType, &newType);
    MPI_Type_commit(&newType);
    return {newType, true};
  }

  public:
  PUML() = default;
  auto operator=(const PUML&) = delete;
  auto operator=(PUML&&) = delete;
  PUML(const PUML&) = delete;
  PUML(PUML&&) = delete;

  virtual ~PUML() {
    delete[] m_originalCells;
    delete[] m_originalVertices;

    for (const auto& i : m_cellData) {
      std::free(i);
    }

    for (const auto& i : m_vertexData) {
      std::free(i);
    }

    for (const auto& i : m_originalVertexData) {
      std::free(i);
    }

#ifdef USE_MPI
    for (size_t i = 0; i < m_cellDataType.size(); ++i) {
      if (m_cellDataTypeDerived[i]) {
        MPI_Type_free(&m_cellDataType[i]);
      }
    }

    for (size_t i = 0; i < m_vertexDataType.size(); ++i) {
      if (m_vertexDataTypeDerived[i]) {
        MPI_Type_free(&m_vertexDataType[i]);
      }
    }
#endif
  }

#ifdef USE_MPI
  void setComm(MPI_Comm comm) { m_comm = comm; }
#endif // USE_MPI

  void open(const char* cellName, const char* vertexName) {
    int rank = 0;
    int procs = 1;
#ifdef USE_MPI
    MPI_Comm_rank(m_comm, &rank);
    MPI_Comm_size(m_comm, &procs);
#endif // USE_MPI

    const auto cellNames = utils::StringUtils::split(cellName, ':');
    if (cellNames.size() != 2) {
      logError() << "Cells name must have the form \"filename:/dataset\"";
    }

    const auto vertexNames = utils::StringUtils::split(vertexName, ':');
    if (vertexNames.size() != 2) {
      logError() << "Vertices name must have the form \"filename:/dataset\"";
    }

    // Open the cell file
    hid_t h5plist = H5Pcreate(H5P_FILE_ACCESS);
    checkH5Err(h5plist);
#ifdef USE_MPI
    checkH5Err(H5Pset_fapl_mpio(h5plist, m_comm, MPI_INFO_NULL));
#endif // USE_MPI

    hid_t h5file = H5Fopen(cellNames[0].c_str(), H5F_ACC_RDONLY, h5plist);
    checkH5Err(h5file);

    // Get cell dataset
    hid_t h5dataset = H5Dopen(h5file, cellNames[1].c_str(), H5P_DEFAULT);
    checkH5Err(h5dataset);

    // Check the size of cell dataset
    hid_t h5space = H5Dget_space(h5dataset);
    checkH5Err(h5space);
    if (H5Sget_simple_extent_ndims(h5space) != 2) {
      logError() << "Cell dataset must have 2 dimensions";
    }
    hsize_t dims[2];
    checkH5Err(H5Sget_simple_extent_dims(h5space, dims, nullptr));
    if (dims[1] != internal::Topology<Topo>::cellvertices()) {
      logError() << "Each cell must have" << internal::Topology<Topo>::cellvertices() << "vertices";
    }

    logInfo() << "Found" << dims[0] << "cells";
    auto cellDistributor = Distributor(dims[0], procs);

    // Read the cells
    m_originalTotalSize[0] = dims[0];
    auto [offsetCells, sizeCells] = cellDistributor.offsetAndSize(rank);
    m_originalSize[0] = sizeCells;

    hsize_t start[2] = {offsetCells, 0};
    hsize_t count[2] = {m_originalSize[0], internal::Topology<Topo>::cellvertices()};

    checkH5Err(H5Sselect_hyperslab(h5space, H5S_SELECT_SET, start, nullptr, count, nullptr));

    hid_t h5memspace = H5Screate_simple(2, count, nullptr);
    checkH5Err(h5memspace);

    hid_t h5alist = H5Pcreate(H5P_DATASET_XFER);
    checkH5Err(h5alist);
#ifdef USE_MPI
    checkH5Err(H5Pset_dxpl_mpio(h5alist, H5FD_MPIO_COLLECTIVE));
#endif // USE_MPI

    m_originalCells = new ocell_t[m_originalSize[0]];
    checkH5Err(H5Dread(h5dataset, H5T_NATIVE_ULONG, h5memspace, h5space, h5alist, m_originalCells));

    // Close cells
    checkH5Err(H5Sclose(h5space));
    checkH5Err(H5Sclose(h5memspace));
    checkH5Err(H5Dclose(h5dataset));
    checkH5Err(H5Fclose(h5file));

    // Open the vertex file
    h5file = H5Fopen(vertexNames[0].c_str(), H5F_ACC_RDONLY, h5plist);
    checkH5Err(h5file);

    // Get vertex dataset
    h5dataset = H5Dopen(h5file, vertexNames[1].c_str(), H5P_DEFAULT);
    checkH5Err(h5dataset);

    // Check the size of vertex dataset
    h5space = H5Dget_space(h5dataset);
    checkH5Err(h5space);
    if (H5Sget_simple_extent_ndims(h5space) != 2) {
      logError() << "Vertex dataset must have 2 dimensions";
    }
    checkH5Err(H5Sget_simple_extent_dims(h5space, dims, nullptr));
    if (dims[1] != internal::Topology<Topo>::dimension()) {
      logError() << "Each vertex must have" << internal::Topology<Topo>::dimension()
                 << "coordinate entries";
    }

    logInfo() << "Found" << dims[0] << "vertices";
    auto vertexDistributor = Distributor(dims[0], procs);

    // Read the vertices
    m_originalTotalSize[1] = dims[0];
    auto [offsetVertices, sizeVertices] = vertexDistributor.offsetAndSize(rank);
    m_originalSize[1] = sizeVertices;

    start[0] = offsetVertices;
    count[0] = m_originalSize[1];
    count[1] = internal::Topology<Topo>::dimension();

    checkH5Err(H5Sselect_hyperslab(h5space, H5S_SELECT_SET, start, nullptr, count, nullptr));

    h5memspace = H5Screate_simple(2, count, nullptr);
    checkH5Err(h5memspace);

    m_originalVertices = new overtex_t[m_originalSize[1]];
    checkH5Err(
        H5Dread(h5dataset, H5T_NATIVE_DOUBLE, h5memspace, h5space, h5alist, m_originalVertices));

    // Close vertices
    checkH5Err(H5Sclose(h5space));
    checkH5Err(H5Sclose(h5memspace));
    checkH5Err(H5Dclose(h5dataset));
    checkH5Err(H5Fclose(h5file));

    // Close other H5 stuff
    checkH5Err(H5Pclose(h5plist));
    checkH5Err(H5Pclose(h5alist));
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
    static_assert(std::is_trivially_copyable_v<T>, "T needs to be trivially copyable");
    static_assert(std::is_trivially_default_constructible_v<T>,
                  "T needs to be trivially default constructible");
    int rank = 0;
    int procs = 1;
#ifdef USE_MPI
    MPI_Comm_rank(m_comm, &rank);
    MPI_Comm_size(m_comm, &procs);
#endif // USE_MPI

    auto cellDistributor = Distributor(m_originalTotalSize[type], procs);
    auto [offset, localSize] = cellDistributor.offsetAndSize(rank);

    size_t elemSize = 1;
    for (auto size : sizes) {
      elemSize *= size;
    }

    void* data = std::malloc(sizeof(T) * localSize * elemSize);
    std::memcpy(data, rawData, sizeof(T) * localSize * elemSize);

    int id = -1;
    switch (type) {
    case CELL: {
      id = m_cellData.size();
      m_cellData.push_back(data);
      m_cellDataSize.push_back(sizeof(T) * elemSize);
#ifdef USE_MPI
      auto [type, derived] = createDatatypeArray(mpiType, elemSize);
      m_cellDataType.push_back(type);
      m_cellDataTypeDerived.push_back(derived);
#endif
    } break;
    case VERTEX: {
      id = m_originalVertexData.size();
      m_originalVertexData.push_back(data);
      m_vertexDataSize.push_back(sizeof(T) * elemSize);
#ifdef USE_MPI
      auto [type, derived] = createDatatypeArray(mpiType, elemSize);
      m_vertexDataType.push_back(type);
      m_vertexDataTypeDerived.push_back(derived);
#endif
    } break;
    }

    return id;
  }

  template <typename T = int>
  auto addData(const char* dataName,
               DataType type,
               const std::vector<size_t>& sizes
#ifdef USE_MPI
               ,
               MPI_Datatype mpiType = MPITypeInfer<T>::type()
#endif
                   ,
               hid_t hdf5Type = HDF5TypeInfer<T>::type()) -> int {
    static_assert(std::is_trivially_copyable_v<T>, "T needs to be trivially copyable");
    static_assert(std::is_trivially_default_constructible_v<T>,
                  "T needs to be trivially default constructible");
    int rank = 0;
    int procs = 1;
#ifdef USE_MPI
    MPI_Comm_rank(m_comm, &rank);
    MPI_Comm_size(m_comm, &procs);
#endif // USE_MPI

    auto cellDistributor = Distributor(m_originalTotalSize[type], procs);
    std::vector<std::string> dataNames = utils::StringUtils::split(dataName, ':');
    if (dataNames.size() != 2) {
      logError() << "Data name must have the form \"filename:/dataset\"";
    }

    // Open the cell file
    hid_t h5plist = H5Pcreate(H5P_FILE_ACCESS);
    checkH5Err(h5plist);
#ifdef USE_MPI
    checkH5Err(H5Pset_fapl_mpio(h5plist, m_comm, MPI_INFO_NULL));
#endif // USE_MPI

    hid_t h5file = H5Fopen(dataNames[0].c_str(), H5F_ACC_RDONLY, h5plist);
    checkH5Err(h5file);

    unsigned long totalSize = m_originalTotalSize[type];

    // Get cell dataset
    hid_t h5dataset = H5Dopen(h5file, dataNames[1].c_str(), H5P_DEFAULT);
    checkH5Err(h5dataset);

    // Check the size of cell dataset
    hid_t h5space = H5Dget_space(h5dataset);
    checkH5Err(h5space);
    const auto dimcount = H5Sget_simple_extent_ndims(h5space);
    if (dimcount != 1 + sizes.size()) {
      logError() << "Dataset must have" << 1 + sizes.size() << "dimension(s), but it has"
                 << dimcount;
    }
    std::vector<hsize_t> dim(1 + sizes.size());
    checkH5Err(H5Sget_simple_extent_dims(h5space, dim.data(), nullptr));
    if (dim[0] != totalSize) {
      logError() << "Dataset has the wrong size:" << dim[0] << "vs." << totalSize;
    }
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      if (dim[i + 1] != sizes[i]) {
        const std::vector<hsize_t> subdims(dim.begin() + 1, dim.end());
        logError() << "Dataset has the wrong subsize:" << subdims << "vs." << sizes;
      }
    }

    // Read the cells
    auto [offset, localSize] = cellDistributor.offsetAndSize(rank);

    size_t elemSize = 1;
    for (auto size : sizes) {
      elemSize *= size;
    }

    std::vector<hsize_t> start = {offset};
    std::vector<hsize_t> count = {localSize};

    for (auto size : sizes) {
      start.push_back(0);
      count.push_back(size);
    }

    checkH5Err(
        H5Sselect_hyperslab(h5space, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr));

    hid_t h5memspace = H5Screate_simple(count.size(), count.data(), nullptr);
    checkH5Err(h5memspace);

    hid_t h5alist = H5Pcreate(H5P_DATASET_XFER);
    checkH5Err(h5alist);
#ifdef USE_MPI
    checkH5Err(H5Pset_dxpl_mpio(h5alist, H5FD_MPIO_COLLECTIVE));
#endif // USE_MPI

    void* data = std::malloc(sizeof(T) * localSize * elemSize);
    checkH5Err(H5Dread(h5dataset, hdf5Type, h5memspace, h5space, h5alist, data));

    // Close data
    checkH5Err(H5Sclose(h5space));
    checkH5Err(H5Sclose(h5memspace));
    checkH5Err(H5Dclose(h5dataset));
    checkH5Err(H5Fclose(h5file));

    // Close other H5 stuff
    checkH5Err(H5Pclose(h5plist));
    checkH5Err(H5Pclose(h5alist));

    int id = -1;
    switch (type) {
    case CELL: {
      id = m_cellData.size();
      m_cellData.push_back(data);
      m_cellDataSize.push_back(sizeof(T) * elemSize);
#ifdef USE_MPI
      auto [type, derived] = createDatatypeArray(mpiType, elemSize);
      m_cellDataType.push_back(type);
      m_cellDataTypeDerived.push_back(derived);
#endif
    } break;
    case VERTEX: {
      id = m_originalVertexData.size();
      m_originalVertexData.push_back(data);
      m_vertexDataSize.push_back(sizeof(T) * elemSize);
#ifdef USE_MPI
      auto [type, derived] = createDatatypeArray(mpiType, elemSize);
      m_vertexDataType.push_back(type);
      m_vertexDataTypeDerived.push_back(derived);
#endif
    } break;
    }
    return id;
  }

  void partition(const int* partition) {
    int rank = 0;
    int procs = 1;
#ifdef USE_MPI
    MPI_Comm_rank(m_comm, &rank);
    MPI_Comm_size(m_comm, &procs);
#endif // USE_MPI

    // Create sorting indices
    auto* indices = new unsigned int[m_originalSize[0]];
    for (unsigned int i = 0; i < m_originalSize[0]; i++) {
      indices[i] = i;
    }

    std::sort(indices, indices + m_originalSize[0], [&](unsigned int i1, unsigned int i2) {
      return partition[i1] < partition[i2];
    });

    // Sort cells
    auto* newCells = new ocell_t[m_originalSize[0]];
    for (unsigned int i = 0; i < m_originalSize[0]; i++) {
      std::memcpy(newCells[i], m_originalCells[indices[i]], sizeof(ocell_t));
    }
    delete[] m_originalCells;
    m_originalCells = newCells;

    // Sort other data
    for (std::size_t j = 0; j < m_cellData.size(); ++j) {
      void* newData = std::malloc(m_originalSize[0] * m_cellDataSize[j]);
      for (unsigned int i = 0; i < m_originalSize[0]; i++) {
        std::memcpy(reinterpret_cast<char*>(newData) + (m_cellDataSize[j] * i),
                    reinterpret_cast<char*>(m_cellData[j]) + (m_cellDataSize[j] * indices[i]),
                    m_cellDataSize[j]);
      }

      std::free(m_cellData[j]);
      m_cellData[j] = newData;
    }

    delete[] indices;

    // Compute exchange info
    int* sendCount = new int[procs];
    memset(sendCount, 0, procs * sizeof(int));

    for (unsigned int i = 0; i < m_originalSize[0]; i++) {
      assert(partition[i] < procs);
      sendCount[partition[i]]++;
    }

    int* recvCount = new int[procs];
#ifdef USE_MPI
    MPI_Alltoall(sendCount, 1, MPI_INT, recvCount, 1, MPI_INT, m_comm);
#else  // USE_MPI
    recvCount[0] = sendCount[0];
#endif // USE_MPI

    int* sDispls = new int[procs];
    int* rDispls = new int[procs];
    sDispls[0] = 0;
    rDispls[0] = 0;
    for (int i = 1; i < procs; i++) {
      sDispls[i] = sDispls[i - 1] + sendCount[i - 1];
      rDispls[i] = rDispls[i - 1] + recvCount[i - 1];
    }

    m_originalSize[0] = rDispls[procs - 1] + recvCount[procs - 1];

#ifdef USE_MPI
    // Exchange the cells
    MPI_Datatype cellType = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(internal::Topology<Topo>::cellvertices(), MPI_UNSIGNED_LONG, &cellType);
    MPI_Type_commit(&cellType);

    newCells = new ocell_t[m_originalSize[0]];
    MPI_Alltoallv(m_originalCells,
                  sendCount,
                  sDispls,
                  cellType,
                  newCells,
                  recvCount,
                  rDispls,
                  cellType,
                  m_comm);

    delete[] m_originalCells;
    m_originalCells = newCells;

    MPI_Type_free(&cellType);

    // Exchange cell data
    for (std::size_t j = 0; j < m_cellData.size(); ++j) {
      void* newData = std::malloc(m_originalSize[0] * m_cellDataSize[j]);
      MPI_Alltoallv(m_cellData[j],
                    sendCount,
                    sDispls,
                    m_cellDataType[j],
                    newData,
                    recvCount,
                    rDispls,
                    m_cellDataType[j],
                    m_comm);

      std::free(m_cellData[j]);
      m_cellData[j] = newData;
    }
#endif // USE_MPI

    delete[] sendCount;
    delete[] recvCount;
    delete[] sDispls;
    delete[] rDispls;
  }

  void generateMesh() {
    int rank = 0;
    int procs = 1;
#ifdef USE_MPI
    MPI_Comm_rank(m_comm, &rank);
    MPI_Comm_size(m_comm, &procs);
#endif // USE_MPI

    auto vertexDistributor = Distributor(m_originalTotalSize[1], procs);
    // Generate a list of vertices we need from other processors
    auto* requiredVertexSets = new std::unordered_set<unsigned long>[procs];
    for (unsigned int i = 0; i < m_originalSize[0]; i++) {
      for (unsigned int j = 0; j < internal::Topology<Topo>::cellvertices(); j++) {
        int proc = vertexDistributor.rankOfEntity(m_originalCells[i][j]);
        assert(proc < procs);

        requiredVertexSets[proc].insert(m_originalCells[i][j]); // Convert to local vid
      }
    }

    // Generate information for requesting vertices
    unsigned int totalVertices = requiredVertexSets[0].size();
    for (int i = 1; i < procs; i++) {
      totalVertices += requiredVertexSets[i].size();
    }

    int* sendCount = new int[procs];

    auto* requiredVertices = new unsigned long[totalVertices];
    unsigned int k = 0;
    for (int i = 0; i < procs; i++) {
      sendCount[i] = requiredVertexSets[i].size();

      for (unsigned long it : requiredVertexSets[i]) {
        assert(k < totalVertices);
        requiredVertices[k++] = it;
      }
    }

    delete[] requiredVertexSets;

    // Exchange required vertex information
    int* recvCount = new int[procs];
#ifdef USE_MPI
    MPI_Alltoall(sendCount, 1, MPI_INT, recvCount, 1, MPI_INT, m_comm);
#else  // USE_MPI
    recvCount[0] = sendCount[0];
#endif // USE_MPI

    int* sDispls = new int[procs];
    int* rDispls = new int[procs];
    sDispls[0] = 0;
    rDispls[0] = 0;
    for (int i = 1; i < procs; i++) {
      sDispls[i] = sDispls[i - 1] + sendCount[i - 1];
      rDispls[i] = rDispls[i - 1] + recvCount[i - 1];
    }

    const unsigned int totalRecv = rDispls[procs - 1] + recvCount[procs - 1];

    auto* distribVertexIds = new unsigned long[totalRecv];
#ifdef USE_MPI
    MPI_Alltoallv(requiredVertices,
                  sendCount,
                  sDispls,
                  MPI_UNSIGNED_LONG,
                  distribVertexIds,
                  recvCount,
                  rDispls,
                  MPI_UNSIGNED_LONG,
                  m_comm);
#endif // USE_MPI

    // Send back vertex coordinates (and other data)
    auto* distribVertices = new overtex_t[totalRecv];
    std::vector<void*> distribData;
    distribData.resize(m_originalVertexData.size());
    for (unsigned int i = 0; i < m_originalVertexData.size(); i++) {
      distribData[i] = std::malloc(totalRecv * m_vertexDataSize[i]);
    }
    auto* sharedRanks = new std::vector<int>[m_originalSize[1]];
    k = 0;
    for (int i = 0; i < procs; i++) {
      for (int j = 0; j < recvCount[i]; j++) {
        assert(k < totalRecv);
        distribVertexIds[k] = vertexDistributor.globalToLocalId(rank, distribVertexIds[k]);

        assert(distribVertexIds[k] < m_originalSize[1]);
        std::memcpy(distribVertices[k], m_originalVertices[distribVertexIds[k]], sizeof(overtex_t));

        // Handle other vertex data
        for (unsigned int l = 0; l < m_originalVertexData.size(); l++) {
          std::memcpy(reinterpret_cast<char*>(distribData[l]) + (m_vertexDataSize[l] * k),
                      reinterpret_cast<char*>(m_originalVertexData[l]) +
                          (m_vertexDataSize[l] * distribVertexIds[k]),
                      m_vertexDataSize[l]);
        }

        // Save all ranks for each vertex
        sharedRanks[distribVertexIds[k]].push_back(i);

        k++;
      }
    }

    auto* recvVertices = new overtex_t[totalVertices];

    for (auto& it : m_vertexData) {
      std::free(it);
    }
    m_vertexData.resize(m_originalVertexData.size());
    for (unsigned int i = 0; i < m_originalVertexData.size(); i++) {
      m_vertexData[i] = std::malloc(totalVertices * m_vertexDataSize[i]);
    }
#ifdef USE_MPI
    MPI_Datatype vertexType = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(internal::Topology<Topo>::dimension(), MPI_DOUBLE, &vertexType);
    MPI_Type_commit(&vertexType);

    MPI_Alltoallv(distribVertices,
                  recvCount,
                  rDispls,
                  vertexType,
                  recvVertices,
                  sendCount,
                  sDispls,
                  vertexType,
                  m_comm);

    MPI_Type_free(&vertexType);

    for (unsigned int i = 0; i < m_originalVertexData.size(); i++) {
      MPI_Alltoallv(distribData[i],
                    recvCount,
                    rDispls,
                    m_vertexDataType[i],
                    m_vertexData[i],
                    sendCount,
                    sDispls,
                    m_vertexDataType[i],
                    m_comm);
    }
#endif // USE_MPI

    delete[] distribVertices;
    for (auto& it : distribData) {
      std::free(it);
    }
    distribData.clear();

    // Send back the number of shared ranks for each vertex
    auto* distNsharedRanks = new unsigned int[totalRecv];
    unsigned int distTotalSharedRanks = 0;
    for (unsigned int i = 0; i < totalRecv; i++) {
      assert(distribVertexIds[i] < m_originalSize[1]);
      distNsharedRanks[i] = sharedRanks[distribVertexIds[i]].size();
      distTotalSharedRanks += distNsharedRanks[i];
    }

    auto* recvNsharedRanks = new unsigned int[totalVertices];
#ifdef USE_MPI
    MPI_Alltoallv(distNsharedRanks,
                  recvCount,
                  rDispls,
                  MPI_UNSIGNED,
                  recvNsharedRanks,
                  sendCount,
                  sDispls,
                  MPI_UNSIGNED,
                  m_comm);
#endif // USE_MPI

    delete[] distNsharedRanks;

    // Setup buffers for exchanging shared ranks
    int* sharedSendCount = new int[procs];
    memset(sharedSendCount, 0, procs * sizeof(int));

    int* distSharedRanks = new int[distTotalSharedRanks];
    k = 0;
    unsigned int l = 0;
    for (int i = 0; i < procs; i++) {
      for (int j = 0; j < recvCount[i]; j++) {
        assert(k < totalRecv);
        assert(l + sharedRanks[distribVertexIds[k]].size() <= distTotalSharedRanks);
        memcpy(&distSharedRanks[l],
               sharedRanks[distribVertexIds[k]].data(),
               sharedRanks[distribVertexIds[k]].size() * sizeof(int));
        l += sharedRanks[distribVertexIds[k]].size();

        sharedSendCount[i] += sharedRanks[distribVertexIds[k]].size();

        k++;
      }
    }

    delete[] distribVertexIds;
    delete[] sharedRanks;
    delete[] recvCount;

    int* sharedRecvCount = new int[procs];
    memset(sharedRecvCount, 0, procs * sizeof(int));

    unsigned int recvTotalSharedRanks = 0;
    k = 0;
    for (int i = 0; i < procs; i++) {
      for (int j = 0; j < sendCount[i]; j++) {
        assert(k < totalVertices);
        recvTotalSharedRanks += recvNsharedRanks[k];
        sharedRecvCount[i] += recvNsharedRanks[k];

        k++;
      }
    }

    delete[] sendCount;

    int* recvSharedRanks = new int[recvTotalSharedRanks];

    sDispls[0] = 0;
    rDispls[0] = 0;
    for (int i = 1; i < procs; i++) {
      sDispls[i] = sDispls[i - 1] + sharedSendCount[i - 1];
      rDispls[i] = rDispls[i - 1] + sharedRecvCount[i - 1];
    }

#ifdef USE_MPI
    MPI_Alltoallv(distSharedRanks,
                  sharedSendCount,
                  sDispls,
                  MPI_INT,
                  recvSharedRanks,
                  sharedRecvCount,
                  rDispls,
                  MPI_INT,
                  m_comm);
#endif // USE_MPI

    delete[] distSharedRanks;
    delete[] sharedSendCount;
    delete[] sharedRecvCount;
    delete[] sDispls;
    delete[] rDispls;

    // Generate the vertex array
    m_vertices.resize(totalVertices);

    k = 0;
    for (unsigned int i = 0; i < totalVertices; i++) {
      m_vertices[i].m_gid = requiredVertices[i];
      memcpy(m_vertices[i].m_coordinate.data(), recvVertices[i], sizeof(overtex_t));
      m_vertices[i].m_sharedRanks.resize(recvNsharedRanks[i] - 1);
      unsigned int l = 0;
      for (unsigned int j = 0; j < recvNsharedRanks[i]; j++) {
        if (recvSharedRanks[k] != rank) {
          m_vertices[i].m_sharedRanks[l++] = recvSharedRanks[k];
        }
        k++;
      }
      std::sort(m_vertices[i].m_sharedRanks.begin(), m_vertices[i].m_sharedRanks.end());
    }

    delete[] requiredVertices;
    delete[] recvVertices;
    delete[] recvSharedRanks;
    delete[] recvNsharedRanks;

    // Construct to g2l map for the vertices
    constructG2L(m_vertices, m_verticesg2l);

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

    std::vector<std::set<unsigned int>> edgeUpward;
    auto* vertexUpward = new std::set<unsigned int>[m_vertices.size()];

    for (unsigned int i = 0; i < m_originalSize[0]; i++) {
      m_cells[i].m_gid = i + cellOffset;

      for (unsigned int j = 0; j < internal::Topology<Topo>::cellvertices(); j++) {
        m_cells[i].m_vertices[j] = m_verticesg2l[m_originalCells[i][j]];
      }

      // Faces
      unsigned int v[internal::Topology<Topo>::dimension()];
      unsigned int faces[internal::Topology<Topo>::cellfaces()];
      for (unsigned int j = 0; j < internal::Topology<Topo>::cellfaces(); ++j) {
        const auto& face = internal::Numbering<Topo>::facevertices()[j];
        for (unsigned int d = 0; d < internal::Topology<Topo>::dimension(); ++d) {
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
        unsigned int w[internal::Topology<Topo>::dimension() - 1];
        for (unsigned int j = 0; j < internal::Topology<Topo>::celledges(); ++j) {
          const auto& edge = internal::Numbering<Topo>::edgevertices()[j];
          const auto& edgeadj = internal::Numbering<Topo>::edgefaces()[j];
          w[0] = m_cells[i].m_vertices[edge[0]];
          w[1] = m_cells[i].m_vertices[edge[1]];
          unsigned int edgeIdx =
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
      for (unsigned int i = 0; i < m_edges.size(); i++) {
        assert(m_edges[i].m_upward.empty());
        m_edges[i].m_upward.resize(edgeUpward[i].size());
        unsigned int j = 0;
        for (auto it = edgeUpward[i].begin(); it != edgeUpward[i].end(); ++it, j++) {
          m_edges[i].m_upward[j] = *it;
        }
      }
      edgeUpward.clear(); // Free memory
    }

    // Set vertex upward information
    for (unsigned int i = 0; i < m_vertices.size(); i++) {
      m_vertices[i].m_upward.resize(vertexUpward[i].size());
      unsigned int j = 0;
      for (auto it = vertexUpward[i].begin(); it != vertexUpward[i].end(); ++it, j++) {
        m_vertices[i].m_upward[j] = *it;
      }
    }
    delete[] vertexUpward;

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
  auto originalCells() const -> const ocell_t* { return m_originalCells; }

  /**
   * @return The original vertices on this rank
   */
  auto originalVertices() const -> const overtex_t* { return m_originalVertices; }

  /**
   * @return Original user cell data
   */
  auto originalCellData(unsigned int index) const -> const void* {
    return cellData(index); // This is the same
  }

  /**
   * @return Original user vertex data
   */
  auto originalVertexData(unsigned int index) const -> const void* {
    if (index >= m_originalVertexData.size()) {
      logError() << "Requested vertex data at index" << index
                 << ", but only so many exist:" << m_originalVertexData.size();
    }
    return m_originalVertexData[index];
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
  auto cellData(unsigned int index) const -> const void* {
    if (index >= m_cellData.size()) {
      logError() << "Requested cell data at index" << index
                 << ", but only so many exist:" << m_cellData.size();
    }
    return m_cellData[index];
  }

  /**
   * @return User vertex data
   */
  auto vertexData(unsigned int index) const -> const void* {
    if (index >= m_vertexData.size()) {
      logError() << "Requested vertex data at index" << index
                 << ", but only so many exist:" << m_vertexData.size();
    }
    return m_vertexData[index];
  }

  /**
   * @param vertexIds A list of local vertex ids
   * @return The local face id for the given set of vertices or <code>-1</code> if
   *  the face does not exist
   */
  auto faceByVertices(unsigned int vertexIds[internal::Topology<Topo>::facevertices()]) const
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
      // Update an old face
      assert(m_faces[lid].m_upward[1] == -1);
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
        assert(allShared[i] != nullptr);
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
    int rank{};
    int procs{};
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

  template <typename TT>
  static void checkH5ErrImpl(TT status, const char* file, int line, int rank) {
    if (status < 0) {
      logError() << utils::nospace << "An HDF5 error occurred in PUML (" << file << ": " << line
                 << ") on rank " << rank;
    }
  }

  public:
  void identify(int dataId) {
    const auto* identifiers = reinterpret_cast<const unsigned long*>(vertexData(dataId));
    for (std::size_t i = 0; i < m_originalSize[0]; ++i) {
      for (std::size_t j = 0; j < internal::Topology<Topo>::cellvertices(); ++j) {
        m_originalCells[i][j] = identifiers[m_cells[i].m_vertices[j]];
      }
    }
  }
};

#undef checkH5Err

/** Convenient typedef for tetrahrdral meshes */
using TETPUML = PUML<TETRAHEDRON>;

} // namespace PUML

#endif // PUML_PUML_H
