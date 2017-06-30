/**
 * @file
 *  This file is part of PUML
 *
 *  For conditions of distribution and use, please see the copyright
 *  notice in the file 'COPYING' at the root directory of this package
 *  and the copyright notice at https://github.com/TUM-I5/PUMGen
 *
 * @copyright 2017 Technische Universitaet Muenchen
 * @author Sebastian Rettenberger <sebastian.rettenberger@tum.de>
 */

#ifndef PUML_H
#define PUML_H

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

#include <hdf5.h>

#include "utils/logger.h"
#include "utils/stringutils.h"

#include "DownElement.h"
#include "Element.h"
#include "Topology.h"
#include "VertexElementMap.h"

namespace PUML
{

enum DataType
{
	CELL = 0,
	VERTEX = 1
};

#define checkH5Err(...) _checkH5Err(__VA_ARGS__, __FILE__, __LINE__)

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
	std::vector<int*> m_cellData;

	/** User vertex data */
	std::vector<int*> m_vertexData;

	/** Original user vertex data */
	std::vector<int*> m_originalVertexData;
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

		for (std::vector<int*>::const_iterator it = m_cellData.begin();
				it != m_cellData.end(); ++it) {
			delete [] *it;
		}

		for (std::vector<int*>::const_iterator it = m_vertexData.begin();
				it != m_vertexData.end(); ++it) {
			delete [] *it;
		}

		for (std::vector<int*>::const_iterator it = m_originalVertexData.begin();
				it != m_originalVertexData.end(); ++it) {
			delete [] *it;
		}
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
		if (cellNames.size() != 2)
			logError() << "Cells name must have the form \"filename:/dataset\"";

		std::vector<std::string> vertexNames = utils::StringUtils::split(vertexName, ':');
		if (vertexNames.size() != 2)
			logError() << "Vertices name must have the form \"filename:/dataset\"";

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
		if (H5Sget_simple_extent_ndims(h5space) != 2)
			logError() << "Cell dataset must have 2 dimensions";
		hsize_t dims[2];
		checkH5Err(H5Sget_simple_extent_dims(h5space, dims, 0L));
		if (dims[1] != internal::Topology<Topo>::cellvertices())
			logError() << "Each cell must have" << internal::Topology<Topo>::cellvertices() << "vertices";

		logInfo(rank) << "Found" << dims[0] << "cells";

		// Read the cells
		m_originalTotalSize[0] = dims[0];
		m_originalSize[0] = (dims[0] + procs - 1) / procs;
		unsigned long offset = m_originalSize[0] * rank;
		m_originalSize[0] = std::min(m_originalSize[0], static_cast<unsigned int>(dims[0] - offset));

		hsize_t start[2] = {offset, 0};
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
		if (H5Sget_simple_extent_ndims(h5space) != 2)
			logError() << "Vertex dataset must have 2 dimensions";
		checkH5Err(H5Sget_simple_extent_dims(h5space, dims, 0L));
		if (dims[1] != 3)
			logError() << "Each vertex must have xyz coordinate";

		logInfo(rank) << "Found" << dims[0] << "vertices";

		// Read the vertices
		m_originalTotalSize[1] = dims[0];
		m_originalSize[1] = (dims[0] + procs - 1) / procs;
		offset = m_originalSize[1] * rank;
		m_originalSize[1] = std::min(m_originalSize[1], static_cast<unsigned int>(dims[0] - offset));

		start[0] = offset;
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

	void addData(const char* dataName, DataType type)
	{
		int rank = 0;
		int procs = 1;
#ifdef USE_MPI
		MPI_Comm_rank(m_comm, &rank);
		MPI_Comm_size(m_comm, &procs);
#endif // USE_MPI

		std::vector<std::string> dataNames = utils::StringUtils::split(dataName, ':');
		if (dataNames.size() != 2)
			logError() << "Data name must have the form \"filename:/dataset\"";

		// Open the cell file
		hid_t h5plist = H5Pcreate(H5P_FILE_ACCESS);
		checkH5Err(h5plist);
#ifdef USE_MPI
		checkH5Err(H5Pset_fapl_mpio(h5plist, m_comm, MPI_INFO_NULL));
#endif // USE_MPI

		hid_t h5file = H5Fopen(dataNames[0].c_str(), H5F_ACC_RDONLY, h5plist);
		checkH5Err(h5file);

		unsigned long totalSize = m_originalTotalSize[type];
		unsigned int localSize = m_originalSize[type];

		// Get cell dataset
		hid_t h5dataset = H5Dopen(h5file, dataNames[1].c_str(), H5P_DEFAULT);
		checkH5Err(h5dataset);

		// Check the size of cell dataset
		hid_t h5space = H5Dget_space(h5dataset);
		checkH5Err(h5space);
		if (H5Sget_simple_extent_ndims(h5space) != 1)
			logError() << "Dataset must have 1 dimension";
		hsize_t dim;
		checkH5Err(H5Sget_simple_extent_dims(h5space, &dim, 0L));
		if (dim != totalSize)
			logError() << "Dataset has the wrong size";

		// Read the cells
		unsigned int maxLocalSize = (totalSize + procs - 1) / procs;
		unsigned long offset = maxLocalSize * rank;

		hsize_t start = offset;
		hsize_t count = localSize;

		checkH5Err(H5Sselect_hyperslab(h5space, H5S_SELECT_SET, &start, 0L, &count, 0L));

		hid_t h5memspace = H5Screate_simple(1, &count, 0L);
		checkH5Err(h5memspace);

		hid_t h5alist = H5Pcreate(H5P_DATASET_XFER);
		checkH5Err(h5alist);
#ifdef USE_MPI
		checkH5Err(H5Pset_dxpl_mpio(h5alist, H5FD_MPIO_COLLECTIVE));
#endif // USE_MPI

		int* data = new int[localSize];
		checkH5Err(H5Dread(h5dataset, H5T_NATIVE_INT, h5memspace, h5space, h5alist, data));

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
			m_cellData.push_back(data);
			break;
		case VERTEX:
			m_originalVertexData.push_back(data);
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
		for (unsigned int i = 0; i < m_originalSize[0]; i++)
			indices[i] = i;

		struct PSort
		{
			const int * const partition;

			PSort(const int* partition) : partition(partition)
			{ }

			bool operator()(unsigned int i1, unsigned int i2) const
			{
				return partition[i1] < partition[i2];
			}
		};
		std::sort(indices, indices+m_originalSize[0], PSort(partition));

		// Sort cells
		ocell_t* newCells = new ocell_t[m_originalSize[0]];
		for (unsigned int i = 0; i < m_originalSize[0]; i++) {
			memcpy(newCells[i], m_originalCells[indices[i]], sizeof(ocell_t));
		}

		// Forget about the unsorted data
		delete [] indices;
		delete [] m_originalCells;
		m_originalCells = newCells;

		// Compute exchange info
		int* sendCount = new int[procs];
		memset(sendCount, 0, procs * sizeof(int));

		for (int i = 0; i < m_originalSize[0]; i++) {
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
		for (std::vector<int*>::iterator it = m_cellData.begin();
				it != m_cellData.end(); ++it) {
			int* newData = new int[m_originalSize[0]];
			MPI_Alltoallv(*it, sendCount, sDispls, MPI_INT,
				newData, recvCount, rDispls, MPI_INT,
				m_comm);

			delete [] *it;
			*it = newData;
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

		// Generate a list of vertices we need from other processors
		unsigned int maxVertices = (m_originalTotalSize[1] + procs - 1) / procs;

		std::unordered_set<unsigned long>* requiredVertexSets = new std::unordered_set<unsigned long>[procs];
		for (unsigned int i = 0; i < m_originalSize[0]; i++) {
			for (unsigned int j = 0; j < internal::Topology<Topo>::cellvertices(); j++) {
				int proc = m_originalCells[i][j] / maxVertices;
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

		// Send back vertex coordinates (an other data)
		overtex_t* distribVertices = new overtex_t[totalRecv];
		std::vector<int*> distribData;
		distribData.resize(m_originalVertexData.size());
		for (unsigned int i = 0; i < m_originalVertexData.size(); i++)
			distribData[i] = new int[totalRecv];
		std::vector<int>* sharedRanks = new std::vector<int>[m_originalSize[1]];
		k = 0;
		for (int i = 0; i < procs; i++) {
			for (unsigned int j = 0; j < recvCount[i]; j++) {
				assert(k < totalRecv);
				distribVertexIds[k] %= maxVertices; // Map to local vertex id
				assert(distribVertexIds[k] < m_originalSize[1]);
				memcpy(distribVertices[k], m_originalVertices[distribVertexIds[k]], sizeof(overtex_t));

				// Handle other vertex data
				for (unsigned int l = 0; l < m_originalVertexData.size(); l++)
					distribData[l][k] = m_originalVertexData[l][distribVertexIds[k]];

				// Save all ranks for each vertex
				sharedRanks[distribVertexIds[k]].push_back(i);

				k++;
			}
		}

		overtex_t* recvVertices = new overtex_t[totalVertices];

		for (std::vector<int*>::iterator it = m_vertexData.begin();
				it != m_vertexData.end(); ++it) {
			delete [] *it;
		}
		m_vertexData.resize(m_originalVertexData.size());
		for (std::vector<int*>::iterator it = m_vertexData.begin();
				it != m_vertexData.end(); ++it) {
			*it = new int[totalVertices];
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
			MPI_Alltoallv(distribData[i], recvCount, rDispls, MPI_INT,
				m_vertexData[i], sendCount, sDispls, MPI_INT,
				m_comm);
		}
#endif // USE_MPI

		delete [] distribVertices;
		for (std::vector<int*>::iterator it = distribData.begin();
				it != distribData.end(); ++it) {
			delete [] *it;
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
			for (unsigned int j = 0; j < recvCount[i]; j++) {
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
			for (unsigned int j = 0; j < sendCount[i]; j++) {
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

			for (unsigned int j = 0; j < internal::Topology<Topo>::cellvertices(); j++)
				m_cells[i].m_vertices[j] = m_verticesg2l[m_originalCells[i][j]];

			// TODO adapt for hex

			// Faces
			unsigned int v[internal::Topology<Topo>::facevertices()];
			unsigned int faces[internal::Topology<Topo>::cellfaces()];
			v[0] = m_cells[i].m_vertices[1];
			v[1] = m_cells[i].m_vertices[0];
			v[2] = m_cells[i].m_vertices[2];
			faces[0] = addFace(m_v2f.add(v), i);
			v[0] = m_cells[i].m_vertices[0];
			v[1] = m_cells[i].m_vertices[1];
			v[2] = m_cells[i].m_vertices[3];
			faces[1] = addFace(m_v2f.add(v), i);
			v[0] = m_cells[i].m_vertices[1];
			v[1] = m_cells[i].m_vertices[2];
			v[2] = m_cells[i].m_vertices[3];
			faces[2] = addFace(m_v2f.add(v), i);
			v[0] = m_cells[i].m_vertices[2];
			v[1] = m_cells[i].m_vertices[0];
			v[2] = m_cells[i].m_vertices[3];
			faces[3] = addFace(m_v2f.add(v), i);

			// Edges
			unsigned int edges[internal::Topology<Topo>::celledges()];
			v[0] = m_cells[i].m_vertices[0];
			v[1] = m_cells[i].m_vertices[1];
			edges[0] = addEdge(edgeUpward, m_v2e.add(v), faces[0], faces[1]);
			v[0] = m_cells[i].m_vertices[1];
			v[1] = m_cells[i].m_vertices[2];
			edges[1] = addEdge(edgeUpward, m_v2e.add(v), faces[0], faces[2]);
			v[0] = m_cells[i].m_vertices[2];
			v[1] = m_cells[i].m_vertices[0];
			edges[2] = addEdge(edgeUpward, m_v2e.add(v), faces[0], faces[3]);
			v[0] = m_cells[i].m_vertices[0];
			v[1] = m_cells[i].m_vertices[3];
			edges[3] = addEdge(edgeUpward, m_v2e.add(v), faces[1], faces[3]);
			v[0] = m_cells[i].m_vertices[1];
			v[1] = m_cells[i].m_vertices[3];
			edges[4] = addEdge(edgeUpward, m_v2e.add(v), faces[1], faces[2]);
			v[0] = m_cells[i].m_vertices[2];
			v[1] = m_cells[i].m_vertices[3];
			edges[5] = addEdge(edgeUpward, m_v2e.add(v), faces[2], faces[3]);

			// Vertices (upward information)
			vertexUpward[m_cells[i].m_vertices[0]].insert(edges[0]);
			vertexUpward[m_cells[i].m_vertices[0]].insert(edges[2]);
			vertexUpward[m_cells[i].m_vertices[0]].insert(edges[3]);
			vertexUpward[m_cells[i].m_vertices[1]].insert(edges[0]);
			vertexUpward[m_cells[i].m_vertices[1]].insert(edges[1]);
			vertexUpward[m_cells[i].m_vertices[1]].insert(edges[4]);
			vertexUpward[m_cells[i].m_vertices[2]].insert(edges[1]);
			vertexUpward[m_cells[i].m_vertices[2]].insert(edges[2]);
			vertexUpward[m_cells[i].m_vertices[2]].insert(edges[5]);
			vertexUpward[m_cells[i].m_vertices[3]].insert(edges[3]);
			vertexUpward[m_cells[i].m_vertices[3]].insert(edges[4]);
			vertexUpward[m_cells[i].m_vertices[3]].insert(edges[5]);
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
			assert(m_vertices[i].m_upward.empty());
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
	const int* originalCellData(unsigned int index) const
	{
		return cellData(index); // This is the same
	}

	/**
	 * @return Original user vertex data
	 */
	const int* originalVertexData(unsigned int index) const
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
	const int* cellData(unsigned int index) const
	{
		return m_cellData[index];
	}

	/**
	 * @return User vertex data
	 */
	const int* vertexData(unsigned int index) const
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
			m_faces[lid].m_upward[1] = plid;
			if (m_faces[lid].m_upward[1] < m_faces[lid].m_upward[0])
				std::swap(m_faces[lid].m_upward[0], m_faces[lid].m_upward[1]);
		} else {
			// New face
			assert(lid == m_faces.size());

			face_t face;
			face.m_upward[0] = plid;
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
			for (std::vector<unsigned int>::const_iterator it2 = it->m_upward.begin();
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
				assert(sharedPos[*it] < nShared[*it]);
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
					assert(sendPos[*it] < nSendGid[*it]);

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
	static void _checkH5Err(TT status, const char* file, int line)
	{
		if (status < 0)
			logError() << utils::nospace << "An HDF5 error occurred in PUML ("
				<< file << ": " << line << ")";
	}
};

#undef checkH5Err

/** Convenient typedef for tetrahrdral meshes */
typedef PUML<TETRAHEDRON> TETPUML;

}

#endif // PUML_H