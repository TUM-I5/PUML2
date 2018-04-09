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

#ifndef PUML_PUML_H
#define PUML_PUML_H
#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <limits>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <array>
#include <utility>
#include <string>
#include <iostream>
#include <fstream>

#include <hdf5.h>

#include "utils/logger.h"
#include "utils/stringutils.h"

#include "DownElement.h"
#include "Element.h"
#include "Topology.h"
#include "FaceMap.h"
#include "EdgeMap.h"
#include "VertexSet.h"

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

    double times[9];


    //Stopwatch mpitime;
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
    //internal::VertexElementMap<internal::Topology<Topo>::facevertices()> m_v2f;

    /** Maps from local vertex ids to local edge ids */
    //internal::VertexElementMap<2> m_v2e;

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
        unsigned long offset = static_cast<unsigned long>(m_originalSize[0]) * rank;
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
        offset = static_cast<unsigned long>(m_originalSize[1]) * rank;
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
        unsigned long offset = static_cast<unsigned long>(maxLocalSize) * rank;

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
        delete [] m_originalCells;
        m_originalCells = newCells;

        // Sort other data
        for (std::vector<int*>::iterator it = m_cellData.begin();
                it != m_cellData.end(); ++it) {
            int* newData = new int[m_originalSize[0]];
            for (unsigned int i = 0; i < m_originalSize[0]; i++) {
                newData[i] = (*it)[indices[i]];
            }

            delete [] *it;
            *it = newData;
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

/*
        int zeroes = 0;
        double nonzeroes = 0;
        double AVGmessageSize = 0;
        int typeSize;
        MPI_Type_size(cellType, &typeSize);
        for(int i = 0; i < procs;i++)
        {
            if(sendCount[i] > 0)
            {
                nonzeroes++;
                AVGmessageSize += sendCount[i] * typeSize;
            }
            else
                zeroes++;
        }
        AVGmessageSize = AVGmessageSize / nonzeroes;
        double totalAVGsizes = 0;
        double totalAVGnonzeroes = 0;
        MPI_Reduce(&AVGmessageSize, &totalAVGsizes, 1, MPI_DOUBLE, MPI_SUM, 0,
                   m_comm);
        MPI_Reduce(&nonzeroes, &totalAVGnonzeroes, 1, MPI_DOUBLE, MPI_SUM, 0,
                   m_comm);
        totalAVGsizes = totalAVGsizes / procs;
        totalAVGnonzeroes = totalAVGnonzeroes / procs;
        if(rank == 0)
            printf("AVG: Processes send on avg messages of size %f to %f different processes\n", totalAVGsizes, totalAVGnonzeroes);

*/
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

        int threads = 64;
        int maxThreads = 128;
        int lessThreads = 32;

        if(omp_get_max_threads() < threads)
            threads = omp_get_max_threads();

        if(omp_get_max_threads() > maxThreads)
            omp_set_num_threads(maxThreads);
        else
            maxThreads = omp_get_max_threads();

        if(omp_get_max_threads() < lessThreads)
            lessThreads = omp_get_max_threads();

        logInfo(rank) << "Starting PUML's steps 3 to 5 with" << omp_get_max_threads() << "cores";
        //logInfo(rank) << "Begin section with " << omp_get_max_threads() << "( " << lessThreads << "/" << threads << "/" << maxThreads << ")" << " cores";
        //Stopwatch totalTime = Stopwatch();
        //totalTime.start();
        //Stopwatch topTime = Stopwatch();
        //topTime.start();
        //Stopwatch stopper = Stopwatch();



        // Generate a list of vertices we need from other processors
        unsigned int maxVertices = (m_originalTotalSize[1] + procs - 1) / procs;

        std::vector<internal::VertexSet> vertexSets;
        for(int i = 0; i < procs; i++)
        {
            internal::VertexSet vs(m_originalSize[0]);
            vertexSets.push_back(vs);
        }
        for(int j = 0; j < procs; j++)
        {
            for(unsigned int i = 0; i < (m_originalSize[0] / LOCK_SIZE) + 1; i++)
            {
                omp_init_lock(&vertexSets[j].lock[i]);
            }
        }

        int* added = new int[maxThreads];
        for(int i = 0; i < maxThreads; i++)
            added[i] = 0;

        #pragma omp parallel for num_threads(maxThreads)
        for (unsigned int i = 0; i < m_originalSize[0]; i++) {
            for (unsigned int j = 0; j < internal::Topology<Topo>::cellvertices(); j++) {
                int proc = m_originalCells[i][j] / maxVertices;
                assert(proc < procs);
                if(vertexSets[proc].add(m_originalCells[i][j]))
                {
                    added[omp_get_thread_num()]++; // Convert to local vid
                }
            }
        }

        unsigned int totalVertices = 0;
        for(int i = 0; i < omp_get_max_threads(); i++)
            totalVertices += added[i];

        delete [] added;

        // Generate information for requesting vertices

        int* sendCount = new int[procs];

        unsigned long* requiredVertices = new unsigned long[totalVertices];
        unsigned int k = 0;

        for (int i = 0; i < procs; i++) {
            int size = 0;
            for (int j = 0; j < m_originalSize[0]; j++) {
                assert(k < totalVertices+1);
                if(vertexSets[i].vertex_table[j] != -1)
                {
                    requiredVertices[k++] = vertexSets[i].vertex_table[j];
                    size++;
                }
            }
            sendCount[i] = size;
        }

        for(int i = 0; i < procs; i++)
        {
            vertexSets[i].clear();
        }
        vertexSets.clear();


        // Exchange required vertex information
        int* recvCount = new int[procs];
#ifdef USE_MPI
        //mpitime = Stopwatch();;
        //mpitime.start();
        MPI_Alltoall(sendCount, 1, MPI_INT, recvCount, 1, MPI_INT, m_comm);
        //mpitime.pause();
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
        
        //mpitime.start();
        MPI_Alltoallv(requiredVertices, sendCount, sDispls, MPI_UNSIGNED_LONG,
            distribVertexIds, recvCount, rDispls, MPI_UNSIGNED_LONG,
            m_comm);
        //mpitime.pause();
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

        //mpitime.start();
        MPI_Alltoallv(distribVertices, recvCount, rDispls, vertexType,
            recvVertices, sendCount, sDispls, vertexType,
            m_comm);

        MPI_Type_free(&vertexType);

        for (unsigned int i = 0; i < m_originalVertexData.size(); i++) {

            MPI_Alltoallv(distribData[i], recvCount, rDispls, MPI_INT,
                m_vertexData[i], sendCount, sDispls, MPI_INT,
                m_comm);
        }
        //mpitime.pause();
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

        //mpitime.start();
        MPI_Alltoallv(distNsharedRanks, recvCount, rDispls, MPI_UNSIGNED,
            recvNsharedRanks, sendCount, sDispls, MPI_UNSIGNED,
            m_comm);
        //mpitime.pause();
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

        //mpitime.start();
        MPI_Alltoallv(distSharedRanks, sharedSendCount, sDispls, MPI_INT,
            recvSharedRanks, sharedRecvCount, rDispls, MPI_INT,
            m_comm);
        //mpitime.pause();
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
       // m_v2f.clear();
        m_faces.clear();
        //m_v2e.clear();

        unsigned long cellOffset = m_originalSize[0];
#ifdef USE_MPI
        //mpitime.start();
        MPI_Scan(MPI_IN_PLACE, &cellOffset, 1, MPI_UNSIGNED_LONG, MPI_SUM, m_comm);
        //mpitime.pause();
#endif // USE_MPI
        cellOffset -= m_originalSize[0];

        logInfo(rank) << "Step 3 successful";



        //times[0] = topTime.pause();
        //topTime.stop();

        //Stopwatch center_time = Stopwatch();
        //center_time.start();

        unsigned int face_table_size = m_originalSize[0] * 4;
        unsigned int edge_table_size = m_originalSize[0] * 6;        

        internal::FaceMap faceMap(face_table_size);
        #pragma omp parallel for num_threads(threads)
        for(unsigned int i = 0; i < face_table_size; i++)
        {
            faceMap.face_table[i].cells[0] = -1;
            faceMap.face_table[i].cells[1] = -1;
        }

        for(unsigned int i = 0; i < (face_table_size / LOCK_SIZE) + 1; i++)
        {
            omp_init_lock(&faceMap.lock[i]);
        }

        //stopper.start();

        int* double_subtract = new int[maxThreads];
        for(int i = 0; i < maxThreads; i++)
            double_subtract[i] = 0;

        #pragma omp parallel for schedule(guided) num_threads(maxThreads)
        for (unsigned int i = 0; i < m_originalSize[0]; i++) {

            m_cells[i].m_gid = i + cellOffset;

            for (unsigned int j = 0; j < internal::Topology<Topo>::cellvertices(); j++)
            {
                m_cells[i].m_vertices[j] = m_verticesg2l[m_originalCells[i][j]];
            }


            unsigned int v[internal::Topology<Topo>::facevertices()];
            unsigned int faces[internal::Topology<Topo>::cellfaces()];

            int sub = 0;
            v[0] = m_cells[i].m_vertices[1];
            v[1] = m_cells[i].m_vertices[0];
            v[2] = m_cells[i].m_vertices[2];
            if(!faceMap.add(v, i))
                sub++;

            v[0] = m_cells[i].m_vertices[0];
            v[1] = m_cells[i].m_vertices[1];
            v[2] = m_cells[i].m_vertices[3];
            if(!faceMap.add(v, i))
                sub++;

            v[0] = m_cells[i].m_vertices[1];
            v[1] = m_cells[i].m_vertices[2];
            v[2] = m_cells[i].m_vertices[3];
            if(!faceMap.add(v, i))
                sub++;

            v[0] = m_cells[i].m_vertices[2];
            v[1] = m_cells[i].m_vertices[0];
            v[2] = m_cells[i].m_vertices[3];
            if(!faceMap.add(v, i))
                sub++;

            if(sub*3 > 6)
                double_subtract[omp_get_thread_num()] += 6;
            else
                double_subtract[omp_get_thread_num()] += sub*3;
        }

        for(int i = 0; i < maxThreads; i++)
        {
            edge_table_size = edge_table_size - double_subtract[i];
        }

        edge_table_size = 6 * edge_table_size;


        //Stopwatch findprime = Stopwatch();
        //findprime.start();
        edge_table_size = NextPrime(edge_table_size);

        //double primetime = findprime.pause();
        //findprime.stop();
        
        //times[1] = stopper.pause();
        //stopper.stop();

        int* distribution_faces = new int[threads];
        for(int i = 0; i < threads; i++)
            distribution_faces[i] = 0;

        #pragma omp parallel for num_threads(threads)
        for(unsigned int i = 0; i < face_table_size; i++)
        {
            distribution_faces[omp_get_thread_num()] += faceMap.face_table[i].cells[0] != -1;
        }

        if(threads > 1)
        {
            for(unsigned int i = 1; i < threads; i++)
            {
                distribution_faces[i] = distribution_faces[i] + distribution_faces[i-1];
            }
        }

        m_faces.clear();
        m_faces.resize(distribution_faces[threads-1]);

        //stopper.start();
        #pragma omp parallel num_threads(threads)
        {
            unsigned int counter;
            if(omp_get_thread_num() == 0)
                counter = 0;
            else
                counter = distribution_faces[omp_get_thread_num()-1];

            #pragma omp for
            for(unsigned int i = 0; i < face_table_size; ++i)
            {
                if(faceMap.face_table[i].cells[0] == -1)
                    continue;

                faceMap.face_table[i].id = counter;

                face_t face;
                face.m_upward[0] = faceMap.face_table[i].cells[0];
                face.m_upward[1] = faceMap.face_table[i].cells[1];
                m_faces[counter] = face;

                counter++;
            }
        }

        delete [] distribution_faces;

        //times[2] = stopper.pause();
        //stopper.stop();
        //stopper.pause();
        //stopper.printTime("Numbering faces and inserting them to target vector (Parallel); ");
        //stopper.stop();
    
        internal::EdgeMap edgeMap(edge_table_size);
        #pragma omp parallel for schedule(guided) num_threads(threads)
        for(unsigned int i = 0; i < edge_table_size; i++)
        {
            edgeMap.edge_table[i].size = 0;
        }

        for(unsigned int i = 0; i < (edge_table_size / LOCK_SIZE) + 1; i++)
        {
            omp_init_lock(&edgeMap.lock[i]);
        }

        omp_lock_t* lock_vertices = new omp_lock_t[(m_vertices.size() / LOCK_SIZE) + 1];

        for(unsigned int i = 0; i < (m_vertices.size() / LOCK_SIZE) + 1; i++)
            omp_init_lock(&(lock_vertices[i]));

        //stopper.start();
        #pragma omp parallel for schedule(guided) num_threads(maxThreads)
        for (unsigned int i = 0; i < m_originalSize[0]; i++) {
            // TODO adapt for hex

            // Find Faces
            unsigned int v[internal::Topology<Topo>::facevertices()];
            unsigned int faces[internal::Topology<Topo>::cellfaces()];
            v[0] = m_cells[i].m_vertices[1];
            v[1] = m_cells[i].m_vertices[0];
            v[2] = m_cells[i].m_vertices[2];
            faces[0] = faceMap.find(v);

            v[0] = m_cells[i].m_vertices[0];
            v[1] = m_cells[i].m_vertices[1];
            v[2] = m_cells[i].m_vertices[3];
            faces[1] = faceMap.find(v);

            v[0] = m_cells[i].m_vertices[1];
            v[1] = m_cells[i].m_vertices[2];
            v[2] = m_cells[i].m_vertices[3];
            faces[2] = faceMap.find(v);

            v[0] = m_cells[i].m_vertices[2];
            v[1] = m_cells[i].m_vertices[0];
            v[2] = m_cells[i].m_vertices[3];
            faces[3] = faceMap.find(v);
            
            // add Edges to HashMap
            unsigned int edges[internal::Topology<Topo>::celledges()];
            unsigned int h;
            v[0] = m_cells[i].m_vertices[0];
            v[1] = m_cells[i].m_vertices[1];
            h = edgeMap.add(v, faces[0], faces[1]);

            if(h != -1)
            {
                omp_set_lock(&(lock_vertices[m_cells[i].m_vertices[0] / LOCK_SIZE]));
                m_vertices[m_cells[i].m_vertices[0]].m_upward.push_back(h);
                omp_unset_lock(&(lock_vertices[m_cells[i].m_vertices[0] / LOCK_SIZE]));
                omp_set_lock(&(lock_vertices[m_cells[i].m_vertices[1] / LOCK_SIZE]));
                m_vertices[m_cells[i].m_vertices[1]].m_upward.push_back(h);
                omp_unset_lock(&(lock_vertices[m_cells[i].m_vertices[1] / LOCK_SIZE]));
            }

            v[0] = m_cells[i].m_vertices[1];
            v[1] = m_cells[i].m_vertices[2];
            h = edgeMap.add(v, faces[0], faces[2]);

            if(h != -1)
            {
                omp_set_lock(&(lock_vertices[m_cells[i].m_vertices[1] / LOCK_SIZE]));
                m_vertices[m_cells[i].m_vertices[1]].m_upward.push_back(h);
                omp_unset_lock(&(lock_vertices[m_cells[i].m_vertices[1] / LOCK_SIZE]));
                omp_set_lock(&(lock_vertices[m_cells[i].m_vertices[2] / LOCK_SIZE]));
                m_vertices[m_cells[i].m_vertices[2]].m_upward.push_back(h);
                omp_unset_lock(&(lock_vertices[m_cells[i].m_vertices[2] / LOCK_SIZE]));
            }

            v[0] = m_cells[i].m_vertices[2];
            v[1] = m_cells[i].m_vertices[0];
            h = edgeMap.add(v, faces[0], faces[3]);

            if(h != -1)
            {
                omp_set_lock(&(lock_vertices[m_cells[i].m_vertices[2] / LOCK_SIZE]));
                m_vertices[m_cells[i].m_vertices[2]].m_upward.push_back(h);
                omp_unset_lock(&(lock_vertices[m_cells[i].m_vertices[2] / LOCK_SIZE]));
                omp_set_lock(&(lock_vertices[m_cells[i].m_vertices[0] / LOCK_SIZE]));
                m_vertices[m_cells[i].m_vertices[0]].m_upward.push_back(h);
                omp_unset_lock(&(lock_vertices[m_cells[i].m_vertices[0] / LOCK_SIZE]));
            }

            v[0] = m_cells[i].m_vertices[0];
            v[1] = m_cells[i].m_vertices[3];
            h = edgeMap.add(v, faces[1], faces[3]);

            if(h != -1)
            {
                omp_set_lock(&(lock_vertices[m_cells[i].m_vertices[0] / LOCK_SIZE]));
                m_vertices[m_cells[i].m_vertices[0]].m_upward.push_back(h);
                omp_unset_lock(&(lock_vertices[m_cells[i].m_vertices[0] / LOCK_SIZE]));
                omp_set_lock(&(lock_vertices[m_cells[i].m_vertices[3] / LOCK_SIZE]));
                m_vertices[m_cells[i].m_vertices[3]].m_upward.push_back(h);
                omp_unset_lock(&(lock_vertices[m_cells[i].m_vertices[3] / LOCK_SIZE]));
            }

            v[0] = m_cells[i].m_vertices[1];
            v[1] = m_cells[i].m_vertices[3];
            h = edgeMap.add(v, faces[1], faces[2]);

            if(h != -1)
            {
                omp_set_lock(&(lock_vertices[m_cells[i].m_vertices[1] / LOCK_SIZE]));
                m_vertices[m_cells[i].m_vertices[1]].m_upward.push_back(h);
                omp_unset_lock(&(lock_vertices[m_cells[i].m_vertices[1] / LOCK_SIZE]));
                omp_set_lock(&(lock_vertices[m_cells[i].m_vertices[3] / LOCK_SIZE]));
                m_vertices[m_cells[i].m_vertices[3]].m_upward.push_back(h);
                omp_unset_lock(&(lock_vertices[m_cells[i].m_vertices[3] / LOCK_SIZE]));
            }

            v[0] = m_cells[i].m_vertices[2];
            v[1] = m_cells[i].m_vertices[3];
            h = edgeMap.add(v, faces[2], faces[3]);

            if(h != -1)
            {
                omp_set_lock(&(lock_vertices[m_cells[i].m_vertices[2] / LOCK_SIZE]));
                m_vertices[m_cells[i].m_vertices[2]].m_upward.push_back(h);
                omp_unset_lock(&(lock_vertices[m_cells[i].m_vertices[2] / LOCK_SIZE]));
                omp_set_lock(&(lock_vertices[m_cells[i].m_vertices[3] / LOCK_SIZE]));
                m_vertices[m_cells[i].m_vertices[3]].m_upward.push_back(h);
                omp_unset_lock(&(lock_vertices[m_cells[i].m_vertices[3] / LOCK_SIZE]));
            }
        }
        //times[3] = stopper.pause();
        //stopper.stop();

        for(unsigned int i = 0; i < (m_vertices.size() / LOCK_SIZE) + 1; i++)
            omp_destroy_lock(&(lock_vertices[i]));

        delete [] lock_vertices;


        faceMap.clear();

        int* distribution_edges = new int[lessThreads];
        for(int i = 0; i < lessThreads; i++)
            distribution_edges[i] = 0;

        #pragma omp parallel for num_threads(lessThreads)
        for(unsigned int i = 0; i < edge_table_size; i++)
        {
            distribution_edges[omp_get_thread_num()] += edgeMap.edge_table[i].size > 0;
        }

        if(lessThreads > 1)
        {
            for(unsigned int i = 1; i < lessThreads; i++)
            {
                distribution_edges[i] = distribution_edges[i] + distribution_edges[i-1];
            }
        }

        //printf("Rank %i, edge_table_size = %i, edges in table = %i\n", rank, edge_table_size, distribution_edges[lessThreads-1]);

        m_edges.clear();
        m_edges.resize(distribution_edges[lessThreads-1]);     

        //stopper.start();
        #pragma omp parallel num_threads(lessThreads)
        {
            unsigned int counter;
            if(omp_get_thread_num() == 0)
                counter = 0;
            else
                counter = distribution_edges[omp_get_thread_num()-1];

            #pragma omp for
            for(unsigned int i = 0; i < edge_table_size; ++i)
            {
                if(edgeMap.edge_table[i].size == 0)
                    continue;
                edgeMap.edge_table[i].id = counter;
                m_edges[counter].m_upward.resize(edgeMap.edge_table[i].size);
                unsigned int j = 0;
                for(j = 0; j < edgeMap.edge_table[i].size && j < CONTAINER_SIZE; j++)
                {
                    m_edges[counter].m_upward[j] = edgeMap.edge_table[i].faces[j];
                }
                if(edgeMap.edge_table[i].size > CONTAINER_SIZE)
                {
                    for (std::set<unsigned int>::const_iterator it = edgeMap.edge_table[i].additionalFaces[0].begin();
                        it != edgeMap.edge_table[i].additionalFaces[0].end(); ++it, j++) {
                        m_edges[counter].m_upward[j] = *it;
                    }
                }
                counter++;
            }
        }
        //times[4] = stopper.pause();
        //stopper.stop();

        delete [] distribution_edges;

        //stopper.start();
        #pragma omp parallel for num_threads(maxThreads)
        for(unsigned int i = 0; i < m_vertices.size(); i++)
        {
            for(unsigned int j = 0; j < m_vertices[i].m_upward.size(); j++)
            {
                m_vertices[i].m_upward[j] = edgeMap.edge_table[m_vertices[i].m_upward[j]].id;
            }
        }
        //times[5] = stopper.pause();
        //stopper.stop();

        edgeMap.clear();
 
        //times[6] = center_time.pause();
        //center_time.stop();

        // VERIFICATION 
        
        /*
        
        std::vector<std::vector<unsigned int>> e_down(m_edges.size(), std::vector<unsigned int>());

        for(unsigned int i = 0; i < m_vertices.size(); i++)
        {
            for(unsigned int j = 0; j < m_vertices[i].m_upward.size(); j++)
            {
                e_down[m_vertices[i].m_upward[j]].push_back(i);
            }
        }

        std::vector<std::vector<unsigned int>> f_down(m_faces.size(), std::vector<unsigned int>());

        std::vector<std::vector<unsigned int>> f_down_down(m_faces.size(), std::vector<unsigned int>());

        for(unsigned int i = 0; i < m_edges.size(); i++)
        {
            for(unsigned int j = 0; j < m_edges[i].m_upward.size(); j++)
            {
                f_down[m_edges[i].m_upward[j]].push_back(i);
            }
        }

        for(unsigned int i = 0; i < f_down.size(); i++)
        {
            f_down_down[i].push_back(e_down[f_down[i][0]][0]);
            f_down_down[i].push_back(e_down[f_down[i][0]][1]);
            if(e_down[f_down[i][1]][0] != f_down_down[i][0] && e_down[f_down[i][1]][0] != f_down_down[i][1])
                f_down_down[i].push_back(e_down[f_down[i][1]][0]);
            else
                f_down_down[i].push_back(e_down[f_down[i][1]][1]);

            f_down_down[i][0] = m_vertices[f_down_down[i][0]].gid();
            f_down_down[i][1] = m_vertices[f_down_down[i][1]].gid();
            f_down_down[i][2] = m_vertices[f_down_down[i][2]].gid();
        }

        std::vector<bool> vertexVisited(m_vertices.size(), true);
        std::vector<bool> double_edges(m_edges.size(), false);
        std::vector<unsigned int> edges_vertices;
        std::string filename = "edge_validation";
        filename += std::to_string(rank);
        std::ofstream edge_validation (filename);
        if(edge_validation.is_open())
        {
            for (unsigned int i = 0; i < m_originalSize[0]; i++)
            {
                unsigned int v = -1;
                for(unsigned int j = 0; j < 4; j++)
                {
                    if(vertexVisited[m_cells[i].m_vertices[j]])
                        vertexVisited[m_cells[i].m_vertices[j]] = false;
                    else
                        continue;
                    edges_vertices.clear();
                    
                    for(unsigned int k = 0; k < m_vertices[m_cells[i].m_vertices[j]].m_upward.size(); k++)
                    {
                        unsigned int edge_id = m_vertices[m_cells[i].m_vertices[j]].m_upward[k];
                        if(m_cells[i].m_vertices[j] == e_down[edge_id][0])
                            v = e_down[edge_id][1];
                        else
                            v = e_down[edge_id][0];

                        v = m_vertices[v].gid();

                        edges_vertices.push_back(v);
                    }

                    std::sort(edges_vertices.begin(), edges_vertices.end());
                    for(unsigned int k = 0; k < edges_vertices.size(); k++)
                    {
                        edge_validation << m_vertices[m_cells[i].m_vertices[j]].gid() << " and " << edges_vertices[k] << "\n";
                    }
                }
            }
            edge_validation.close();
        }
        std::vector<bool> vertexVisited2(m_vertices.size(), true);
        std::vector<std::pair<std::pair<unsigned int, unsigned int>, std::pair<unsigned int, unsigned int>>> faces_vertices;
        std::vector<bool> double_faces(m_faces.size(), false);
        filename = "face_validation";
        filename += std::to_string(rank);
        std::ofstream face_validation (filename);
        if(face_validation.is_open())
        {
            for (unsigned int i = 0; i < m_originalSize[0]; i++)
            {
                for(unsigned int j = 0; j < 4; j++)
                {
                    if(vertexVisited2[m_cells[i].m_vertices[j]])
                        vertexVisited2[m_cells[i].m_vertices[j]] = false;
                    else
                        continue;
                    faces_vertices.clear();
                    
                    for(unsigned int k = 0; k < m_vertices[m_cells[i].m_vertices[j]].m_upward.size(); k++)
                    {
                        unsigned int edge_id = m_vertices[m_cells[i].m_vertices[j]].m_upward[k];
                        for(unsigned int l = 0; l < m_edges[edge_id].m_upward.size(); l++)
                        {
                            unsigned int face_id = m_edges[edge_id].m_upward[l];
                            if(f_down_down[face_id][0] == m_vertices[m_cells[i].m_vertices[j]].gid())
                            {
                                if(f_down_down[face_id][1] > f_down_down[face_id][2])
                                    faces_vertices.push_back({{f_down_down[face_id][2], f_down_down[face_id][1]}, {m_faces[face_id].m_upward[0], m_faces[face_id].m_upward[1]}});
                                else
                                    faces_vertices.push_back({{f_down_down[face_id][1], f_down_down[face_id][2]}, {m_faces[face_id].m_upward[0], m_faces[face_id].m_upward[1]}});
                            }
                            else if(f_down_down[face_id][1] == m_vertices[m_cells[i].m_vertices[j]].gid())
                            {
                                if(f_down_down[face_id][0] > f_down_down[face_id][2])
                                    faces_vertices.push_back({{f_down_down[face_id][2], f_down_down[face_id][0]}, {m_faces[face_id].m_upward[0], m_faces[face_id].m_upward[1]}});
                                else
                                    faces_vertices.push_back({{f_down_down[face_id][0], f_down_down[face_id][2]}, {m_faces[face_id].m_upward[0], m_faces[face_id].m_upward[1]}});
                            }
                            else if(f_down_down[face_id][2] == m_vertices[m_cells[i].m_vertices[j]].gid())
                            {
                                if(f_down_down[face_id][1] > f_down_down[face_id][0])
                                    faces_vertices.push_back({{f_down_down[face_id][0], f_down_down[face_id][1]}, {m_faces[face_id].m_upward[0], m_faces[face_id].m_upward[1]}});
                                else
                                    faces_vertices.push_back({{f_down_down[face_id][1], f_down_down[face_id][0]}, {m_faces[face_id].m_upward[0], m_faces[face_id].m_upward[1]}});
                            }
                        }
                        
                    }

                    std::sort(faces_vertices.begin(), faces_vertices.end());
                    for(unsigned int k = 0; k < faces_vertices.size(); k = k + 2)
                    {
                        face_validation << m_vertices[m_cells[i].m_vertices[j]].gid() << " and " << faces_vertices[k].first.first << " and " << faces_vertices[k].first.second << "  ->  "  << faces_vertices[k].second.first << " and " << faces_vertices[k].second.second << "\n";
                    }
                }
            }
            face_validation.close();
        }
        */
       
        logInfo(rank) << "Step 4 successful";

        //Stopwatch bottomTime = Stopwatch();
        //bottomTime.start();

        // Generate shared information and global ids for edges
        generatedSharedAndGID<edge_t, vertex_t, 2>(m_edges, m_vertices, maxThreads);

        // Generate shared information and global ids for faces
        generatedSharedAndGID<face_t, edge_t, internal::Topology<Topo>::faceedges()>(m_faces, m_edges, maxThreads);

        logInfo(rank) << "Step 5 successful";
        //times[7] = bottomTime.pause();
        //bottomTime.stop();

        //times[8] = totalTime.pause();
        //totalTime.stop();
    }

    // from https://stackoverflow.com/questions/30052316/find-next-prime-number-algorithm
    static bool IsPrime(int number)
    {

        if (number == 2 || number == 3)
            return true;

        if (number % 2 == 0 || number % 3 == 0)
            return false;

        int divisor = 6;
        while (divisor * divisor - 2 * divisor + 1 <= number)
        {

            if (number % (divisor - 1) == 0)
                return false;

            if (number % (divisor + 1) == 0)
                return false;

            divisor += 6;

        }

        return true;

    }
    // from https://stackoverflow.com/questions/30052316/find-next-prime-number-algorithm
    static int NextPrime(int a)
    {

        while (!IsPrime(++a)) 
        { }
        return a;

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
     * Generates the shared information and global ids from the downward elements
     *
     * @param elements The elements for which the data should be generated
     * @param down The downward elements
     * @tparam N The number of downward elements
     */
    template<typename E, typename D, unsigned int N>
    void generatedSharedAndGID(std::vector<E> &elements, const std::vector<D> &down, int maxThreads)
    {
#ifdef USE_MPI
        // Collect all shared ranks for each element and downward gids
        const std::vector<int> ** allShared = new const std::vector<int>*[elements.size() * N];
        memset(allShared, 0, elements.size() * N * sizeof(std::vector<int>*));
        unsigned long* downward = new unsigned long[elements.size() * N];
        unsigned int* downPos = new unsigned int[elements.size()];
        memset(downPos, 0, elements.size() * sizeof(unsigned int));

        omp_lock_t* lock_elements = new omp_lock_t[(elements.size() / LOCK_SIZE) + 1];

        for(unsigned int i = 0; i < (elements.size() / LOCK_SIZE) + 1; i++)
            omp_init_lock(&(lock_elements[i]));
        
        #pragma omp parallel for schedule(guided) num_threads(maxThreads)
        for (typename std::vector<D>::const_iterator it = down.begin();
                it != down.end(); ++it) {
            for (std::vector<int>::const_iterator it2 = it->m_upward.begin();
                    it2 != it->m_upward.end(); ++it2) {

                omp_set_lock(&(lock_elements[*it2 / LOCK_SIZE]));

                assert(downPos[*it2] < N);
                allShared[*it2 * N + downPos[*it2]] = &it->m_sharedRanks;
                downward[*it2 * N + downPos[*it2]] = it->m_gid;
                downPos[*it2]++;
                omp_unset_lock(&(lock_elements[*it2 / LOCK_SIZE]));
            }
        }


        for(unsigned int i = 0; i < (elements.size() / LOCK_SIZE) + 1; i++)
            omp_destroy_lock(&(lock_elements[i]));

        delete [] lock_elements;

        delete [] downPos;


        // Create the intersection of the shared ranks and update the elements
        assert(N >= 2);

        #pragma omp parallel for num_threads(maxThreads)
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

        std::vector<std::vector<int>> nSharedThread(maxThreads, std::vector<int>(procs));

        #pragma omp parallel for schedule(guided) num_threads(maxThreads)
        for (typename std::vector<E>::const_iterator it = elements.begin();
                it != elements.end(); ++it) {
            for (std::vector<int>::const_iterator it2 = it->m_sharedRanks.begin();
                    it2 != it->m_sharedRanks.end(); ++it2) {
                nSharedThread[omp_get_thread_num()][*it2]++;
            }
        }

        for(int i = 0; i < maxThreads; i++)
            for(int j = 0; j < procs; j++)
                nShared[j] += nSharedThread[i][j];

        int *nRecvShared = new int[procs];

        //mpitime.start();
        MPI_Alltoall(nShared, 1, MPI_INT, nRecvShared, 1, MPI_INT, m_comm);

        //mpitime.pause();

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

        //mpitime.start();
        MPI_Alltoallv(sendShared, nShared, sDispls, type,
            recvShared, nRecvShared, rDispls, type,
            m_comm);
        //mpitime.pause();

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
        #pragma omp parallel for schedule(guided) num_threads(maxThreads)
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
        //mpitime.start();
        MPI_Scan(MPI_IN_PLACE, &gidOffset, 1, MPI_UNSIGNED_LONG, MPI_SUM, m_comm);
        //mpitime.pause();
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

        //mpitime.start();
        MPI_Alltoallv(sendGid, nSendGid, sDispls, MPI_UNSIGNED_LONG,
            recvGid, nRecvGid, rDispls, MPI_UNSIGNED_LONG,
            m_comm);

        MPI_Alltoallv(sendDGid, nSendGid, sDispls, type,
            recvDGid, nRecvGid, rDispls, type,
            m_comm);

        //printf("Time spend in MPI stuff: %f", mpitime.pause());
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

#endif // PUML_PUML_H
