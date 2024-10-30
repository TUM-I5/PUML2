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
#include <type_traits>
#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include <algorithm>
#include <cassert>
#include <limits>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdlib>

#include <hdf5.h>

#include "utils/logger.h"
#include "utils/stringutils.h"

#include "DownElement.h"
#include "Element.h"
#include "Numbering.h"
#include "Topology.h"
#include "VertexElementMap.h"

namespace PUML
{

enum DataType
{
	CELL = 0,
	VERTEX = 1
};

/**
 * Distributes a number of mesh entities (i.e. elements or vertices) to  a given number of ranks
 * For E entities and R ranks, the first E%R ranks will read E/R+1 entities.
 * The remaining ranks will read E/R entities.
 */
class Distributor {
  public: 
  Distributor() = default;
  Distributor(unsigned long newNumEntities, unsigned long newNumRanks) : numEntities(newNumEntities),
    numRanks(newNumRanks),
    entitiesPerRank(numEntities / numRanks),
    missingEntities(numEntities % numRanks) {
    assert(numEntities > numRanks); 
  }

  /**
   * Gives the offset and size of data where the rank should read data.
   */
  std::pair<unsigned long, unsigned long> offsetAndSize(unsigned long rank) {
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
  unsigned long rankOfEntity(unsigned long globalId) {
    assert(globalId < numEntities);
    unsigned long rank = 0;
    if (globalId < missingEntities * (entitiesPerRank + 1)) {
      rank = globalId / (entitiesPerRank + 1);
    } else {
      rank = (globalId - missingEntities * (entitiesPerRank + 1)) / entitiesPerRank + missingEntities;
    }
    assert(rank < numRanks);
    return rank;
  }

  /**
   * Gives the local id of a mesh entity for the given globalId.
   */
  unsigned long globalToLocalId(unsigned long rank, unsigned long globalId) {
    assert(globalId < numEntities);
    auto[offset, size] = offsetAndSize(rank);
    assert (globalId >= offset);
    assert (globalId < offset + size);
    return globalId - offset;
  }

  private:
  const unsigned long numEntities;
  const unsigned long numRanks;
  const unsigned long entitiesPerRank;
  const unsigned long missingEntities;
};

#define checkH5Err(...) _checkH5Err(__VA_ARGS__, __FILE__, __LINE__, rank)

/**
 * @todo Handle non-MPI case correct
 */
template<TopoType Topo>
class PUML
{
public:
	/** The cell type from the file */
	typedef unsigned long ocell_t[internal::Topology<Topo>::cellvertices()];

	/** The vertex type from the file */
	typedef double overtex_t[3];

	/** Internal cell type */
	typedef Cell<Topo> cell_t;

	/** Internal face type */
	typedef Face face_t;

	/** Internal edge type */
	typedef Edge edge_t;

	/** Internal vertex type */
	typedef Vertex vertex_t;
private:
#ifdef USE_MPI
	MPI_Comm m_comm;
#endif // USE_MPI

	/** The original cells from the file */
	ocell_t* m_originalCells;

	/** The original vertices from the file */
	overtex_t* m_originalVertices;

	/** The original number of cells/vertices on each node */
	unsigned int m_originalSize[2];

	/** The original number of total cells/vertices */
	unsigned long m_originalTotalSize[2];

	typedef std::unordered_map<unsigned long, unsigned int> g2l_t;

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

	std::pair<MPI_Datatype, bool> createDatatypeArray(MPI_Datatype baseType, std::size_t elemSize) {
		if (elemSize == 1) {
			return {baseType, false};
		}
		MPI_Datatype newType;
		MPI_Type_contiguous(elemSize, baseType, &newType);
		MPI_Type_commit(&newType);
		return {newType, true};
	}

public:
	PUML() :
#ifdef USE_MPI
		m_comm(MPI_COMM_WORLD),
#endif // USE_MPI
		m_originalCells(0L),
		m_originalVertices(0L)
	{ }

	virtual ~PUML()
	{
		delete [] m_originalCells;
		delete [] m_originalVertices;

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
	void setComm(MPI_Comm comm)
	{
		m_comm = comm;
	}
#endif // USE_MPI

	void open(const char* cellName, const char* vertexName)
	{
		int rank = 0;
		int procs = 1;
#ifdef USE_MPI
		MPI_Comm_rank(m_comm, &rank);
		MPI_Comm_size(m_comm, &procs);
#endif // USE_MPI

		std::vector<std::string> cellNames = utils::StringUtils::split(cellName, ':');
		if (cellNames.size() != 2) {
			logError() << "Cells name must have the form \"filename:/dataset\"";
		}

		std::vector<std::string> vertexNames = utils::StringUtils::split(vertexName, ':');
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
		checkH5Err(H5Sget_simple_extent_dims(h5space, dims, 0L));
		if (dims[1] != internal::Topology<Topo>::cellvertices()) {
			logError() << "Each cell must have" << internal::Topology<Topo>::cellvertices() << "vertices";
		}

		logInfo(rank) << "Found" << dims[0] << "cells";
		auto cellDistributor = Distributor(dims[0], procs);

		// Read the cells
		m_originalTotalSize[0] = dims[0];
		auto[offsetCells, sizeCells] = cellDistributor.offsetAndSize(rank);
		m_originalSize[0] = sizeCells;

		hsize_t start[2] = {offsetCells, 0};
		hsize_t count[2] = {m_originalSize[0], internal::Topology<Topo>::cellvertices()};

		checkH5Err(H5Sselect_hyperslab(h5space, H5S_SELECT_SET, start, 0L, count, 0L));

		hid_t h5memspace = H5Screate_simple(2, count, 0L);
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
		checkH5Err(H5Sget_simple_extent_dims(h5space, dims, 0L));
		if (dims[1] != 3) {
			logError() << "Each vertex must have xyz coordinate";
		}

		logInfo(rank) << "Found" << dims[0] << "vertices";
		auto vertexDistributor = Distributor(dims[0], procs);

		// Read the vertices
		m_originalTotalSize[1] = dims[0];
		auto[offsetVertices, sizeVertices] = vertexDistributor.offsetAndSize(rank);
		m_originalSize[1] = sizeVertices;

		start[0] = offsetVertices;
		count[0] = m_originalSize[1]; count[1] = 3;

		checkH5Err(H5Sselect_hyperslab(h5space, H5S_SELECT_SET, start, 0L, count, 0L));

		h5memspace = H5Screate_simple(2, count, 0L);
		checkH5Err(h5memspace);

		m_originalVertices = new overtex_t[m_originalSize[1]];
		checkH5Err(H5Dread(h5dataset, H5T_NATIVE_DOUBLE, h5memspace, h5space, h5alist, m_originalVertices));

		// Close vertices
		checkH5Err(H5Sclose(h5space));
		checkH5Err(H5Sclose(h5memspace));
		checkH5Err(H5Dclose(h5dataset));
		checkH5Err(H5Fclose(h5file));

		// Close other H5 stuff
		checkH5Err(H5Pclose(h5plist));
		checkH5Err(H5Pclose(h5alist));
	}

    template<typename T>
	void addDataArray(const T* rawData, DataType type, const std::vector<size_t>& sizes
#ifdef USE_MPI
		, MPI_Datatype mpiType = MPITypeInfer<T>::type()
#endif
	)
	{
		static_assert(std::is_trivially_copyable_v<T>, "T needs to be trivially copyable");
		static_assert(std::is_trivially_default_constructible_v<T>, "T needs to be trivially default constructible");
		int rank = 0;
		int procs = 1;
#ifdef USE_MPI
		MPI_Comm_rank(m_comm, &rank);
		MPI_Comm_size(m_comm, &procs);
#endif // USE_MPI

		auto cellDistributor = Distributor(m_originalTotalSize[type], procs);
		auto[offset, localSize] = cellDistributor.offsetAndSize(rank);

		size_t elemSize = 1;
		for (auto size : sizes) {
			elemSize *= size;
		}

		void* data = std::malloc(sizeof(T) * localSize * elemSize);
        std::memcpy(data, rawData, sizeof(T) * localSize * elemSize);
		switch (type) {
		case CELL:
		{
			m_cellData.push_back(data);
			m_cellDataSize.push_back(sizeof(T) * elemSize);
#ifdef USE_MPI
			auto [type, derived] = createDatatypeArray(mpiType, elemSize);
			m_cellDataType.push_back(type);
			m_cellDataTypeDerived.push_back(derived);
#endif
		}
			break;
		case VERTEX:
		{
			m_originalVertexData.push_back(data);
			m_vertexDataSize.push_back(sizeof(T) * elemSize);
#ifdef USE_MPI
			auto [type, derived] = createDatatypeArray(mpiType, elemSize);
			m_vertexDataType.push_back(type);
			m_vertexDataTypeDerived.push_back(derived);
#endif
		}
			break;
		}
	}

	template<typename T = int>
	void addData(const char* dataName, DataType type, const std::vector<size_t>& sizes
#ifdef USE_MPI
		, MPI_Datatype mpiType = MPITypeInfer<T>::type()
#endif
		, hid_t hdf5Type = HDF5TypeInfer<T>::type()
	)
	{
		static_assert(std::is_trivially_copyable_v<T>, "T needs to be trivially copyable");
		static_assert(std::is_trivially_default_constructible_v<T>, "T needs to be trivially default constructible");
		int rank = 0;
		int procs = 1;
#ifdef USE_MPI
		MPI_Comm_rank(m_comm, &rank);
		MPI_Comm_size(m_comm, &procs);
#endif // USE_MPI

		auto cellDistributor = Distributor(m_originalTotalSize[0], procs);
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
			logError() << "Dataset must have" << 1 + sizes.size() << "dimension(s), but it has" << dimcount;
		}
		std::vector<hsize_t> dim(1 + sizes.size());
		checkH5Err(H5Sget_simple_extent_dims(h5space, dim.data(), 0L));
		if (dim[0] != totalSize) {
			logError() << "Dataset has the wrong size:" << dim[0] << "vs." << totalSize;
		}
		for (std::size_t i = 0; i < sizes.size(); ++i) {
			if (dim[i + 1] != sizes[i]) {
				std::vector<hsize_t> subdims(dim.begin() + 1, dim.end());
				logError() << "Dataset has the wrong subsize:" << subdims << "vs." << sizes;
			}
		}

		// Read the cells
		auto[offset, localSize] = cellDistributor.offsetAndSize(rank);

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

		checkH5Err(H5Sselect_hyperslab(h5space, H5S_SELECT_SET, start.data(), 0L, count.data(), 0L));

		hid_t h5memspace = H5Screate_simple(count.size(), count.data(), 0L);
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

		switch (type) {
		case CELL:
		{
			m_cellData.push_back(data);
			m_cellDataSize.push_back(sizeof(T) * elemSize);
#ifdef USE_MPI
			auto [type, derived] = createDatatypeArray(mpiType, elemSize);
			m_cellDataType.push_back(type);
			m_cellDataTypeDerived.push_back(derived);
#endif
		}
			break;
		case VERTEX:
		{
			m_originalVertexData.push_back(data);
			m_vertexDataSize.push_back(sizeof(T) * elemSize);
#ifdef USE_MPI
			auto [type, derived] = createDatatypeArray(mpiType, elemSize);
			m_vertexDataType.push_back(type);
			m_vertexDataTypeDerived.push_back(derived);
#endif
		}
			break;
		}
	}

	void partition(int* partition)
	{
		int rank = 0;
		int procs = 1;
#ifdef USE_MPI
		MPI_Comm_rank(m_comm, &rank);
		MPI_Comm_size(m_comm, &procs);
#endif // USE_MPI

		// Create sorting indices
		unsigned int* indices = new unsigned int[m_originalSize[0]];
		for (unsigned int i = 0; i < m_originalSize[0]; i++) {
			indices[i] = i;
		}

		std::sort(indices, indices+m_originalSize[0], [&](unsigned int i1, unsigned int i2)
			{
				return partition[i1] < partition[i2];
			});

		// Sort cells
		ocell_t* newCells = new ocell_t[m_originalSize[0]];
		for (unsigned int i = 0; i < m_originalSize[0]; i++) {
			std::memcpy(newCells[i], m_originalCells[indices[i]], sizeof(ocell_t));
		}
		delete [] m_originalCells;
		m_originalCells = newCells;

		// Sort other data
		for (std::size_t j = 0; j < m_cellData.size(); ++j) {
			void* newData = std::malloc(m_originalSize[0] * m_cellDataSize[j]);
			for (unsigned int i = 0; i < m_originalSize[0]; i++) {
				std::memcpy(reinterpret_cast<char*>(newData) + m_cellDataSize[j] * i, reinterpret_cast<char*>(m_cellData[j]) + m_cellDataSize[j] * indices[i], m_cellDataSize[j]);
			}

			std::free(m_cellData[j]);
			m_cellData[j] = newData;
		}

		delete [] indices;

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
#else // USE_MPI
		recvCount[0] = sendCount[0];
#endif // USE_MPI

		int *sDispls = new int[procs];
		int *rDispls = new int[procs];
		sDispls[0] = 0;
		rDispls[0] = 0;
		for (int i = 1; i < procs; i++) {
			sDispls[i] = sDispls[i-1] + sendCount[i-1];
			rDispls[i] = rDispls[i-1] + recvCount[i-1];
		}

		m_originalSize[0] = rDispls[procs-1] + recvCount[procs-1];

#ifdef USE_MPI
		// Exchange the cells
		MPI_Datatype cellType;
		MPI_Type_contiguous(internal::Topology<Topo>::cellvertices(), MPI_UNSIGNED_LONG, &cellType);
		MPI_Type_commit(&cellType);

		newCells = new ocell_t[m_originalSize[0]];
		MPI_Alltoallv(m_originalCells, sendCount, sDispls, cellType,
			newCells, recvCount, rDispls, cellType,
			m_comm);

		delete [] m_originalCells;
		m_originalCells = newCells;

		MPI_Type_free(&cellType);

		// Exchange cell data
		for (std::size_t j = 0; j < m_cellData.size(); ++j) {
			void* newData = std::malloc(m_originalSize[0] * m_cellDataSize[j]);
			MPI_Alltoallv(m_cellData[j], sendCount, sDispls, m_cellDataType[j],
				newData, recvCount, rDispls, m_cellDataType[j],
				m_comm);

			std::free(m_cellData[j]);
			m_cellData[j] = newData;
		}
#endif // USE_MPI

		delete [] sendCount;
		delete [] recvCount;
		delete [] sDispls;
		delete [] rDispls;
	}

	void generateMesh()
	{
		int rank = 0;
		int procs = 1;
#ifdef USE_MPI
		MPI_Comm_rank(m_comm, &rank);
		MPI_Comm_size(m_comm, &procs);
#endif // USE_MPI

		auto vertexDistributor = Distributor(m_originalTotalSize[1], procs);
		// Generate a list of vertices we need from other processors
		std::unordered_set<unsigned long>* requiredVertexSets = new std::unordered_set<unsigned long>[procs];
		for (unsigned int i = 0; i < m_originalSize[0]; i++) {
			for (unsigned int j = 0; j < internal::Topology<Topo>::cellvertices(); j++) {
				int proc = vertexDistributor.rankOfEntity(m_originalCells[i][j]);
				assert(proc < procs);

				requiredVertexSets[proc].insert(m_originalCells[i][j]); // Convert to local vid
			}
		}

		// Generate information for requesting vertices
		unsigned int totalVertices = requiredVertexSets[0].size();
		for (int i = 1; i < procs; i++)
			totalVertices += requiredVertexSets[i].size();

		int* sendCount = new int[procs];

		unsigned long* requiredVertices = new unsigned long[totalVertices];
		unsigned int k = 0;
		for (int i = 0; i < procs; i++) {
			sendCount[i] = requiredVertexSets[i].size();

			for (std::unordered_set<unsigned long>::const_iterator it = requiredVertexSets[i].begin();
					it != requiredVertexSets[i].end(); ++it) {
				assert(k < totalVertices);
				requiredVertices[k++] = *it;
			}
		}

		delete [] requiredVertexSets;

		// Exchange required vertex information
		int* recvCount = new int[procs];
#ifdef USE_MPI
		MPI_Alltoall(sendCount, 1, MPI_INT, recvCount, 1, MPI_INT, m_comm);
#else // USE_MPI
		recvCount[0] = sendCount[0];
#endif // USE_MPI

		int *sDispls = new int[procs];
		int *rDispls = new int[procs];
		sDispls[0] = 0;
		rDispls[0] = 0;
		for (int i = 1; i < procs; i++) {
			sDispls[i] = sDispls[i-1] + sendCount[i-1];
			rDispls[i] = rDispls[i-1] + recvCount[i-1];
		}

		unsigned int totalRecv = rDispls[procs-1] + recvCount[procs-1];

		unsigned long* distribVertexIds = new unsigned long[totalRecv];
#ifdef USE_MPI
		MPI_Alltoallv(requiredVertices, sendCount, sDispls, MPI_UNSIGNED_LONG,
			distribVertexIds, recvCount, rDispls, MPI_UNSIGNED_LONG,
			m_comm);
#endif // USE_MPI

		// Send back vertex coordinates (and other data)
		overtex_t* distribVertices = new overtex_t[totalRecv];
		std::vector<void*> distribData;
		distribData.resize(m_originalVertexData.size());
		for (unsigned int i = 0; i < m_originalVertexData.size(); i++) {
			distribData[i] = std::malloc(totalRecv * m_vertexDataSize[i]);
		}
		std::vector<int>* sharedRanks = new std::vector<int>[m_originalSize[1]];
		k = 0;
		for (int i = 0; i < procs; i++) {
			for (int j = 0; j < recvCount[i]; j++) {
				assert(k < totalRecv);
				distribVertexIds[k] = vertexDistributor.globalToLocalId(rank, distribVertexIds[k]);

				assert(distribVertexIds[k] < m_originalSize[1]);
				std::memcpy(distribVertices[k], m_originalVertices[distribVertexIds[k]], sizeof(overtex_t));

				// Handle other vertex data
				for (unsigned int l = 0; l < m_originalVertexData.size(); l++) {
					std::memcpy(reinterpret_cast<char*>(distribData[l]) + m_vertexDataSize[l] * k, reinterpret_cast<char*>(m_originalVertexData[l]) + m_vertexDataSize[l] * distribVertexIds[k], m_vertexDataSize[l]);
				}

				// Save all ranks for each vertex
				sharedRanks[distribVertexIds[k]].push_back(i);

				k++;
			}
		}

		overtex_t* recvVertices = new overtex_t[totalVertices];

		for (auto& it : m_vertexData) {
			std::free(it);
		}
		m_vertexData.resize(m_originalVertexData.size());
		for (unsigned int i = 0; i < m_originalVertexData.size(); i++) {
			m_vertexData[i] = std::malloc(totalVertices * m_vertexDataSize[i]);
		}
#ifdef USE_MPI
		MPI_Datatype vertexType;
		MPI_Type_contiguous(3, MPI_DOUBLE, &vertexType);
		MPI_Type_commit(&vertexType);

		MPI_Alltoallv(distribVertices, recvCount, rDispls, vertexType,
			recvVertices, sendCount, sDispls, vertexType,
			m_comm);

		MPI_Type_free(&vertexType);

		for (unsigned int i = 0; i < m_originalVertexData.size(); i++) {
			MPI_Alltoallv(distribData[i], recvCount, rDispls, m_vertexDataType[i],
				m_vertexData[i], sendCount, sDispls, m_vertexDataType[i],
				m_comm);
		}
#endif // USE_MPI

		delete [] distribVertices;
		for (auto& it : distribData) {
			std::free(it);
		}
		distribData.clear();

		// Send back the number of shared ranks for each vertex
		unsigned int* distNsharedRanks = new unsigned int[totalRecv];
		unsigned int distTotalSharedRanks = 0;
		for (unsigned int i = 0; i < totalRecv; i++) {
			assert(distribVertexIds[i] < m_originalSize[1]);
			distNsharedRanks[i] = sharedRanks[distribVertexIds[i]].size();
			distTotalSharedRanks += distNsharedRanks[i];
		}

		unsigned int* recvNsharedRanks = new unsigned int[totalVertices];
#ifdef USE_MPI
		MPI_Alltoallv(distNsharedRanks, recvCount, rDispls, MPI_UNSIGNED,
			recvNsharedRanks, sendCount, sDispls, MPI_UNSIGNED,
			m_comm);
#endif // USE_MPI

		delete [] distNsharedRanks;

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
				memcpy(&distSharedRanks[l], &sharedRanks[distribVertexIds[k]][0], sharedRanks[distribVertexIds[k]].size() * sizeof(int));
				l += sharedRanks[distribVertexIds[k]].size();

				sharedSendCount[i] += sharedRanks[distribVertexIds[k]].size();

				k++;
			}
		}

		delete [] distribVertexIds;
		delete [] sharedRanks;
		delete [] recvCount;

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

		delete [] sendCount;

		int* recvSharedRanks = new int[recvTotalSharedRanks];

		sDispls[0] = 0;
		rDispls[0] = 0;
		for (int i = 1; i < procs; i++) {
			sDispls[i] = sDispls[i-1] + sharedSendCount[i-1];
			rDispls[i] = rDispls[i-1] + sharedRecvCount[i-1];
		}

#ifdef USE_MPI
		MPI_Alltoallv(distSharedRanks, sharedSendCount, sDispls, MPI_INT,
			recvSharedRanks, sharedRecvCount, rDispls, MPI_INT,
			m_comm);
#endif // USE_MPI

		delete [] distSharedRanks;
		delete [] sharedSendCount;
		delete [] sharedRecvCount;
		delete [] sDispls;
		delete [] rDispls;

		// Generate the vertex array
		m_vertices.resize(totalVertices);

		k = 0;
		for (unsigned int i = 0; i < totalVertices; i++) {
			m_vertices[i].m_gid = requiredVertices[i];
			memcpy(m_vertices[i].m_coordinate, recvVertices[i], sizeof(overtex_t));
			m_vertices[i].m_sharedRanks.resize(recvNsharedRanks[i]-1);
			unsigned int l = 0;
			for (unsigned int j = 0; j < recvNsharedRanks[i]; j++) {
				if (recvSharedRanks[k] != rank)
					m_vertices[i].m_sharedRanks[l++] = recvSharedRanks[k];
				k++;
			}
			std::sort(m_vertices[i].m_sharedRanks.begin(), m_vertices[i].m_sharedRanks.end());
		}

		delete [] requiredVertices;
		delete [] recvVertices;
		delete [] recvSharedRanks;
		delete [] recvNsharedRanks;

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

		std::vector<std::set<unsigned int> > edgeUpward;
		std::set<unsigned int>* vertexUpward = new std::set<unsigned int>[m_vertices.size()];

		for (unsigned int i = 0; i < m_originalSize[0]; i++) {
			m_cells[i].m_gid = i + cellOffset;

			for (unsigned int j = 0; j < internal::Topology<Topo>::cellvertices(); j++) {
				m_cells[i].m_vertices[j] = m_verticesg2l[m_originalCells[i][j]];
			}

			// Faces
			unsigned int v[internal::Topology<Topo>::facevertices()];
			unsigned int faces[internal::Topology<Topo>::cellfaces()];
			for (unsigned int j = 0; j < internal::Topology<Topo>::cellfaces(); ++j) {
				const auto& face = internal::Numbering<Topo>::facevertices()[j];
				v[0] = m_cells[i].m_vertices[face[0]];
				v[1] = m_cells[i].m_vertices[face[1]];
				v[2] = m_cells[i].m_vertices[face[2]];
				faces[j] = addFace(m_v2f.add(v), i);
			}

			// Edges + Vertex upward information
			for (unsigned int j = 0; j < internal::Topology<Topo>::celledges(); ++j) {
				const auto& edge = internal::Numbering<Topo>::edgevertices()[j];
				const auto& edgeadj = internal::Numbering<Topo>::edgefaces()[j];
				v[0] = m_cells[i].m_vertices[edge[0]];
				v[1] = m_cells[i].m_vertices[edge[1]];
				unsigned int edgeIdx = addEdge(edgeUpward, m_v2e.add(v), faces[edgeadj[0]], faces[edgeadj[1]]);
				vertexUpward[m_cells[i].m_vertices[edge[0]]].insert(edgeIdx);
				vertexUpward[m_cells[i].m_vertices[edge[1]]].insert(edgeIdx);
			}
		}

		// Create edges
		m_edges.clear();
		m_edges.resize(edgeUpward.size());
		for (unsigned int i = 0; i < m_edges.size(); i++) {
			assert(m_edges[i].m_upward.empty());
			m_edges[i].m_upward.resize(edgeUpward[i].size());
			unsigned int j = 0;
			for (std::set<unsigned int>::const_iterator it = edgeUpward[i].begin();
					it != edgeUpward[i].end(); ++it, j++) {
				m_edges[i].m_upward[j] = *it;
			}
		}
		edgeUpward.clear(); // Free memory

		// Set vertex upward information
		for (unsigned int i = 0; i < m_vertices.size(); i++) {
			m_vertices[i].m_upward.resize(vertexUpward[i].size());
			unsigned int j = 0;
			for (std::set<unsigned int>::const_iterator it = vertexUpward[i].begin();
					it != vertexUpward[i].end(); ++it, j++) {
				m_vertices[i].m_upward[j] = *it;
			}
		}
		delete [] vertexUpward;

		// Generate shared information and global ids for edges
		generatedSharedAndGID<edge_t, vertex_t, 2>(m_edges, m_vertices);

		// Generate shared information and global ids for faces
		generatedSharedAndGID<face_t, edge_t, internal::Topology<Topo>::faceedges()>(m_faces, m_edges);
	}

	/**
	 * @return The total number of all cells within the mesh.
	 */
	unsigned int numTotalCells() const
	{
		return m_originalTotalSize[0];
	}

	/**
	 * @return The number of original cells on this rank
	 *
	 * @note This value can change when {@link partition()} is called
	 */
	unsigned int numOriginalCells() const
	{
		return m_originalSize[0];
	}

	/**
	 * @return The number of original vertices on this rank
	 */
	unsigned int numOriginalVertices() const
	{
		return m_originalSize[1];
	}

	/**
	 * @return The original cells on this rank
	 *
	 * @note The pointer gets invalid when {@link partition()} is called
	 */
	const ocell_t* originalCells() const
	{
		return m_originalCells;
	}

	/**
	 * @return The original vertices on this rank
	 */
	const overtex_t* originalVertices() const
	{
		return m_originalVertices;
	}

	/**
	 * @return Original user cell data
	 */
	const void* originalCellData(unsigned int index) const
	{
		return cellData(index); // This is the same
	}

	/**
	 * @return Original user vertex data
	 */
	const void* originalVertexData(unsigned int index) const
	{
		return m_originalVertexData[index];
	}

	/**
	 * @return The cells of the mesh
	 */
	const std::vector<cell_t>& cells() const
	{
		return m_cells;
	}

	/**
	 * @return The faces of the mesh
	 */
	const std::vector<face_t>& faces() const
	{
		return m_faces;
	}

	/**
	 * @return The edges of the mesh
	 */
	const std::vector<edge_t>& edges() const
	{
		return m_edges;
	}

	/**
	 * @return The vertices of the mesh
	 */
	const std::vector<vertex_t>& vertices() const
	{
		return m_vertices;
	}

	/**
	 * @return User cell data
	 */
	const void* cellData(unsigned int index) const
	{
		return m_cellData[index];
	}

	/**
	 * @return User vertex data
	 */
	const void* vertexData(unsigned int index) const
	{
		return m_vertexData[index];
	}

	/**
	 * @param vertexIds A list of local vertex ids
	 * @return The local face id for the given set of vertices or <code>-1</code> if
	 *  the face does not exist
	 */
	int faceByVertices(unsigned int vertexIds[internal::Topology<Topo>::facevertices()]) const
	{
		return m_v2f.find(vertexIds);
	}

#ifdef USE_MPI
	/**
	 * @return The MPI communicator used by this class. (this field only exists, if MPI is enabled)
	 */
	const MPI_Comm& comm() const
	{
		return m_comm;
	}
#endif

private:
	/**
	 * Add a face but only if it does not exist yet
	 *
	 * @param lid The local id of the face
	 * @param plid The local id of the parent
	 * @return The local id of the face
	 */
	unsigned int addFace(unsigned int lid, unsigned int plid)
	{
		if (lid < m_faces.size()) {
			// Update an old face
			assert(m_faces[lid].m_upward[1] == -1);
			m_faces[lid].m_upward[1] = plid;
			if (m_faces[lid].m_upward[1] < m_faces[lid].m_upward[0])
				std::swap(m_faces[lid].m_upward[0], m_faces[lid].m_upward[1]);
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
	template<typename E, typename D, unsigned int N>
	void generatedSharedAndGID(std::vector<E> &elements, const std::vector<D> &down)
	{
#ifdef USE_MPI
		// Collect all shared ranks for each element and downward gids
		const std::vector<int> ** allShared = new const std::vector<int>*[elements.size() * N];
		memset(allShared, 0, elements.size() * N * sizeof(std::vector<int>*));
		unsigned long* downward = new unsigned long[elements.size() * N];
		unsigned int* downPos = new unsigned int[elements.size()];
		memset(downPos, 0, elements.size() * sizeof(unsigned int));

		for (typename std::vector<D>::const_iterator it = down.begin();
				it != down.end(); ++it) {
			for (std::vector<int>::const_iterator it2 = it->m_upward.begin();
					it2 != it->m_upward.end(); ++it2) {
				assert(downPos[*it2] < N);
				allShared[*it2 * N + downPos[*it2]] = &it->m_sharedRanks;
				downward[*it2 * N + downPos[*it2]] = it->m_gid;
				downPos[*it2]++;
			}
		}

		delete [] downPos;

		// Create the intersection of the shared ranks and update the elements
		assert(N >= 2);
		for (unsigned int i = 0; i < elements.size(); i++) {
			assert(allShared[i*N]);
			assert(allShared[i*N+1]);

			std::set_intersection(allShared[i*N]->begin(), allShared[i*N]->end(),
				allShared[i*N + 1]->begin(), allShared[i*N + 1]->end(),
				std::back_inserter(elements[i].m_sharedRanks));

			std::vector<int> buffer;
			for (unsigned int j = 2; j < N; j++) {
				buffer.clear();

				assert(allShared[i*N+j]);
				std::set_intersection(elements[i].m_sharedRanks.begin(), elements[i].m_sharedRanks.end(),
					allShared[i*N + j]->begin(), allShared[i*N + j]->end(),
					std::back_inserter(buffer));

				std::swap(elements[i].m_sharedRanks, buffer);
			}
		}

		delete [] allShared;

		// Eliminate false positves
		int rank, procs;
		MPI_Comm_rank(m_comm, &rank);
		MPI_Comm_size(m_comm, &procs);

		int* nShared = new int[procs];
		memset(nShared, 0, procs * sizeof(int));
		for (typename std::vector<E>::const_iterator it = elements.begin();
				it != elements.end(); ++it) {
			for (std::vector<int>::const_iterator it2 = it->m_sharedRanks.begin();
					it2 != it->m_sharedRanks.end(); ++it2) {
				nShared[*it2]++;
			}
		}

		int *nRecvShared = new int[procs];
		MPI_Alltoall(nShared, 1, MPI_INT, nRecvShared, 1, MPI_INT, m_comm);

		int *sDispls = new int[procs];
		int *rDispls = new int[procs];
		sDispls[0] = 0;
		rDispls[0] = 0;
		for (int i = 1; i < procs; i++) {
			sDispls[i] = sDispls[i-1] + nShared[i-1];
			rDispls[i] = rDispls[i-1] + nRecvShared[i-1];
		}

		unsigned int totalShared = sDispls[procs-1] + nShared[procs-1];

		unsigned int* sharedPos = new unsigned int[procs];
		memset(sharedPos, 0, procs * sizeof(unsigned int));

		unsigned long* sendShared = new unsigned long[totalShared * N];

		for (unsigned int i = 0; i < elements.size(); i++) {
			for (std::vector<int>::const_iterator it = elements[i].m_sharedRanks.begin();
					it != elements[i].m_sharedRanks.end(); ++it) {
				assert(sharedPos[*it] < static_cast<unsigned>(nShared[*it]));
				memcpy(&sendShared[(sDispls[*it]+sharedPos[*it])*N], &downward[i * N],
					N * sizeof(unsigned long));
				sharedPos[*it]++;
			}
		}

		delete [] sharedPos;

		unsigned int totalRecvShared = rDispls[procs-1] + nRecvShared[procs-1];

		unsigned long* recvShared = new unsigned long[totalRecvShared * N];

		MPI_Datatype type;
		MPI_Type_contiguous(N, MPI_UNSIGNED_LONG, &type);
		MPI_Type_commit(&type);

		MPI_Alltoallv(sendShared, nShared, sDispls, type,
			recvShared, nRecvShared, rDispls, type,
			m_comm);

		delete [] nShared;
		delete [] sendShared;

		std::unordered_set<internal::DownElement<N>, internal::DownElementHash<N> >* hashedElements
			= new std::unordered_set<internal::DownElement<N>, internal::DownElementHash<N> >[procs];

		unsigned int k = 0;
		for (int i = 0; i < procs; i++) {
			assert(i != rank || nRecvShared[i] == 0);
			for (int j = 0; j < nRecvShared[i]; j++) {
				assert(k < totalRecvShared);
				assert(hashedElements[i].find(&recvShared[k*N]) == hashedElements[i].end());
				hashedElements[i].emplace(&recvShared[k*N]);
				k++;
			}
		}

		delete [] nRecvShared;
		delete [] recvShared;

		unsigned int e = 0;
		for (unsigned int i = 0; i < elements.size(); i++) {
			internal::DownElement<N> delem(&downward[i * N]);

			std::vector<int>::iterator it = elements[i].m_sharedRanks.begin();
			while (it != elements[i].m_sharedRanks.end()) {
				if (hashedElements[*it].find(delem) == hashedElements[*it].end()) {
					if ((rank == 0 && *it == 3) || (rank == 3 && *it == 0))
						e++;
					it = elements[i].m_sharedRanks.erase(it);
				} else {
					++it;
				}
			}
		}

		delete [] hashedElements;

		// Count owned elements
		unsigned int owned = 0;
		for (typename std::vector<E>::const_iterator it = elements.begin();
				it != elements.end(); ++it) {
			if (it->m_sharedRanks.empty() || it->m_sharedRanks[0] > rank)
				owned++;
		}

		// Get global id offset
		unsigned long gidOffset = owned;
		MPI_Scan(MPI_IN_PLACE, &gidOffset, 1, MPI_UNSIGNED_LONG, MPI_SUM, m_comm);
		gidOffset -= owned;

		// Set global ids for owned elements and count the number of elements we need to forward
		int* nSendGid = new int[procs];
		memset(nSendGid, 0, procs * sizeof(int));
		int* nRecvGid = new int[procs];
		memset(nRecvGid, 0, procs * sizeof(int));
		for (typename std::vector<E>::iterator it = elements.begin();
				it != elements.end(); ++it) {
			if (it->m_sharedRanks.empty() || it->m_sharedRanks[0] > rank) {
				it->m_gid = gidOffset++;

				for (std::vector<int>::const_iterator it2 = it->m_sharedRanks.begin();
						it2 != it->m_sharedRanks.end(); ++it2) {
					nSendGid[*it2]++;
				}
			} else {
				it->m_gid = std::numeric_limits<unsigned long>::max();

				if (!it->m_sharedRanks.empty())
					nRecvGid[it->m_sharedRanks[0]]++;
			}
		}

		// Compute exchange offsets
		sDispls[0] = 0;
		rDispls[0] = 0;
		for (int i = 1; i < procs; i++) {
			sDispls[i] = sDispls[i-1] + nSendGid[i-1];
			rDispls[i] = rDispls[i-1] + nRecvGid[i-1];
		}

		unsigned int totalSendGid = sDispls[procs-1] + nSendGid[procs-1];
		unsigned int totalRecvGid = rDispls[procs-1] + nRecvGid[procs-1];

		// Collect send data
		unsigned int* sendPos = new unsigned int[procs];
		memset(sendPos, 0, procs * sizeof(unsigned int));

		unsigned long* sendGid = new unsigned long[totalSendGid];
		unsigned long* sendDGid = new unsigned long[totalSendGid * N];
		for (unsigned int i = 0; i < elements.size(); i++) {
			if (elements[i].m_sharedRanks.empty() || elements[i].m_sharedRanks[0] > rank) {
				for (std::vector<int>::const_iterator it = elements[i].m_sharedRanks.begin();
						it != elements[i].m_sharedRanks.end(); ++it) {
					assert(sendPos[*it] < static_cast<unsigned>(nSendGid[*it]));

					sendGid[sDispls[*it] + sendPos[*it]] = elements[i].m_gid;
					memcpy(&sendDGid[(sDispls[*it] + sendPos[*it])*N], &downward[i*N], N*sizeof(unsigned long));
					sendPos[*it]++;
				}
			}
		}

		delete [] sendPos;

		// Exchange cell data
		unsigned long* recvGid = new unsigned long[totalRecvGid];
		unsigned long* recvDGid = new unsigned long[totalRecvGid*N];

		MPI_Alltoallv(sendGid, nSendGid, sDispls, MPI_UNSIGNED_LONG,
			recvGid, nRecvGid, rDispls, MPI_UNSIGNED_LONG,
			m_comm);

		MPI_Alltoallv(sendDGid, nSendGid, sDispls, type,
			recvDGid, nRecvGid, rDispls, type,
			m_comm);

		MPI_Type_free(&type);

		delete [] sendGid;
		delete [] sendDGid;
		delete [] nSendGid;
		delete [] nRecvGid;
		delete [] sDispls;
		delete [] rDispls;

		// Create a hash map from the received elements
		std::unordered_map<internal::DownElement<N>, unsigned long, internal::DownElementHash<N> > dg2g;
		for (unsigned int i = 0; i < totalRecvGid; i++) {
			dg2g.emplace(&recvDGid[i*N], recvGid[i]);
		}

		delete [] recvGid;
		delete [] recvDGid;

		// Assign gids
		for (unsigned int i = 0; i < elements.size(); i++) {
			if (!elements[i].m_sharedRanks.empty() && elements[i].m_sharedRanks[0] < rank) {
				assert(elements[i].m_gid == std::numeric_limits<unsigned long>::max());

				internal::DownElement<N> delem(&downward[i * N]);
				typename std::unordered_map<internal::DownElement<N>, unsigned long, internal::DownElementHash<N> >::const_iterator it
					= dg2g.find(delem);
				assert(it != dg2g.end());

				elements[i].m_gid = it->second;
			}
		}

		delete [] downward;
#endif // USE_MPI
	}

private:
	/**
	 * Add an edge if it does not exist yet
	 *
	 * @param edgeUpward The storage for upward information
	 * @param lid The local id of the edge
	 * @param plid1 The first parent
	 * @param plid2 The second parent
	 * @return The local id of the edge
	 */
	static unsigned int addEdge(std::vector<std::set<unsigned int> > &edgeUpward,
		unsigned int lid, unsigned int plid1, unsigned int plid2)
	{
		if (lid >= edgeUpward.size()) {
			assert(lid == edgeUpward.size());
			edgeUpward.push_back(std::set<unsigned int>());
		}
		edgeUpward[lid].insert(plid1);
		edgeUpward[lid].insert(plid2);

		return lid;
	}

	/**
	 * Constructs the global -> local map for an element array
	 */
	template<typename TT>
	static void constructG2L(const std::vector<TT> &elements, g2l_t &g2lMap)
	{
		g2lMap.clear();

		unsigned int i = 0;
		for (typename std::vector<TT>::const_iterator it = elements.begin();
				it != elements.end(); ++it, i++) {
			assert(g2lMap.find(it->m_gid) == g2lMap.end());
			g2lMap[it->m_gid] = i;
		}
	}

	template<typename TT>
	static void _checkH5Err(TT status, const char* file, int line, int rank)
	{
		if (status < 0) {
			logError() << utils::nospace << "An HDF5 error occurred in PUML ("
				<< file << ": " << line << ") on rank " << rank;
		}
	}
};

#undef checkH5Err

/** Convenient typedef for tetrahrdral meshes */
typedef PUML<TETRAHEDRON> TETPUML;

}

#endif // PUML_PUML_H
