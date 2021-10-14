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

#ifndef PUML_PARTITIONMETHIS_H
#define PUML_PARTITIONMETHIS_H

#ifdef USE_MPI
#include <mpi.h>
#endif  // USE_MPI

#include <parmetis.h>
#include <tuple>

#include "Topology.h"

namespace PUML {

template <TopoType Topo>
class PartitionMetis {
 public:
  /** The cell type */
  typedef unsigned long cell_t[internal::Topology<Topo>::cellvertices()];

 private:
#ifdef USE_MPI
  MPI_Comm m_comm;
#endif  // USE_MPI

  const cell_t* const m_cells;
  const idx_t m_numCells;

  int rank = 0;
  int procs = 0;

  idx_t* xadj = nullptr;
  idx_t* adjncy = nullptr;
  idx_t* elemdist = nullptr;

 public:
  PartitionMetis(const cell_t* cells, unsigned int numCells)
      :
#ifdef USE_MPI
        m_comm(MPI_COMM_WORLD),
#endif  // USE_MPI
        m_cells(cells),
        m_numCells(numCells) {
    MPI_Comm_rank(m_comm, &rank);
    MPI_Comm_size(m_comm, &procs);
    elemdist = new idx_t[procs + 1];
  }

  ~PartitionMetis() {
    METIS_Free(xadj);
    METIS_Free(adjncy);
    delete[] elemdist;
  }

#ifdef USE_MPI
  void setComm(MPI_Comm comm) { m_comm = comm; }
#endif  // USE_MPI

#ifdef USE_MPI
  std::tuple<idx_t*, idx_t*> getGraph() {

    if (xadj == nullptr && adjncy == nullptr) {
      generateGraphFromMesh();
    }

    return {xadj, adjncy};
  }
#endif

#ifdef USE_MPI
  void generateGraphFromMesh() {
    assert((xadj == nullptr && adjncy == nullptr) ||
           (xadj != nullptr && adjncy != nullptr));

    if (xadj == nullptr && adjncy == nullptr) {
      MPI_Allgather(const_cast<idx_t*>(&m_numCells), 1, IDX_T, elemdist, 1,
                    IDX_T, m_comm);

      idx_t sum = 0;
      for (int i = 0; i < procs; i++) {
        idx_t e = elemdist[i];
        elemdist[i] = sum;
        sum += e;
      }
      elemdist[procs] = sum;

      idx_t* eptr = new idx_t[m_numCells + 1];
      idx_t* eind =
        new idx_t[m_numCells * internal::Topology<Topo>::cellvertices()];
      unsigned long m = 0;
      for (idx_t i = 0; i < m_numCells; i++) {
        eptr[i] = i * internal::Topology<Topo>::cellvertices();

        for (unsigned int j = 0; j < internal::Topology<Topo>::cellvertices();
             j++) {
          m = std::max(m, m_cells[i][j]);
          eind[i * internal::Topology<Topo>::cellvertices() + j] =
            m_cells[i][j];
        }
      }
      eptr[m_numCells] = m_numCells * internal::Topology<Topo>::cellvertices();

      idx_t numflag = 0;
      idx_t ncommonnodes = 3;  // TODO adapt for hex
      idx_t nparts = procs;

      ParMETIS_V3_Mesh2Dual(elemdist, eptr, eind, &numflag, ncommonnodes, &xadj,
                            &adjncy, &m_comm);

      delete[] elemdist;
      delete[] eptr;
      delete[] eind;
    }
  }
#endif

#ifdef USE_MPI
  /**
   * @param partition An array of size <code>numCells</code> which
   *  will contain the partition for each cells
   * @param vertexWeights Weight for each vertex
   * @param nWeightsPerVertex Number of weights per vertex
   * @param nodeWeights Weight for each node
   * @param imbalance The allowed imbalance
   */
  void partition(int* partition, const int* vertexWeights = nullptr,
                 int nWeightsPerVertex = 1, const double* nodeWeights = nullptr,
                 double* edgeWeights = nullptr, size_t edgeCount = 0,
                 const double* imbalance = 1.05) {

    if (xadj == nullptr && adjncy == nullptr) {
      generateGraphFromMesh();
    }
    // else it was already generated before

    idx_t nparts = procs;
    idx_t ncon = nWeightsPerVertex;
    idx_t numflag = 0;
    idx_t ncommonnodes = 3;

    real_t* tpwgts = new real_t[nparts * ncon];
    if (nodeWeights != nullptr) {
      for (idx_t i = 0; i < nparts; i++) {
        for (idx_t j = 0; j < ncon; ++j) {
          tpwgts[i * ncon + j] = nodeWeights[i];
        }
      }
    } else {
      for (idx_t i = 0; i < nparts * ncon; i++) {
        tpwgts[i] = static_cast<real_t>(1.) / nparts;
      }
    }

    idx_t* elmwgt = nullptr;
    if (vertexWeights != nullptr) {
      elmwgt = new idx_t[m_numCells * ncon];
      for (idx_t cell = 0; cell < m_nemCells; ++cell) {
        for (idx_t j = 0; j < m_numCells; j++) {
          elmwgt[ncon * cell + j] =
            static_cast<idx_t>(vertexWeights[ncon * cell + j]);
        }
      }
    }

    idx_t* edgewgt = nullptr;
    if (edgeWeights != nullptr) {
      edgewgt = new idx_t[edgeCount];
      for (idx_t i = 0; i < edgeCount; i++) {
        edgewgt[i] = static_cast<idx_t>(edgeWeights[i]);
      }
    }

    real_t* ubvec = new real_t[ncon];
    for (idx_t i = 0; i < ncon; ++i) {
      ubvec[i] = imbalance[i];
    }
    idx_t edgecut;
    idx_t options[3] = {1, 0, METIS_RANDOM_SEED};

    idx_t* part = new idx_t[m_numCells];

    idx_t wgtflag = 0;

    if (edgeWeights == nullptr && nodeWeights == nullptr) {
      wgtflag = 0;
    } else if (edgeWeights != nullptr && nodeWeights != nullptr) {
      wgtflag = 3;
    } else if (edgeWeights == nullptr && nodeWeights != nullptr) {
      wgtflag = 2;
    } else {
      wgtflag = 1;
    }

    ParMETIS_V3_PartKway(elemdist, xadj, adjncy, elmwgt, edgewgt, &wgtflag,
                         &numflag, &ncon, &nparts, tpwgts, ubvec, options,
                         &edgecut, part, &m_comm);

    /*
      ParMETIS_V3_PartMeshKway(elemdist, eptr, eind, elmwgt, &wgtflag,
       &numflag, &ncon, &ncommonnodes, &nparts, tpwgts, ubvec, options,
       &edgecut, part, &m_comm);
    */

    delete[] tpwgts;
    delete[] ubvec;
    delete[] elmwgt;
    delete[] edgewgt;

    for (idx_t i = 0; i < m_numCells; i++)
      partition[i] = part[i];

    delete[] part;
  }
#endif  // USE_MPI

 private:
  static const int METIS_RANDOM_SEED = 42;
};

/** Convenient typedef for tetrahrdral meshes */
typedef PartitionMetis<TETRAHEDRON> TETPartitionMetis;

}  // namespace PUML

#endif  // PUML_PARTITIONMETHIS_H
