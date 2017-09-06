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
#endif // USE_MPI

#include <parmetis.h>

#include "Topology.h"

namespace PUML
{

template<TopoType Topo>
class PartitionMetis
{
public:
	/** The cell type */
	typedef unsigned long cell_t[internal::Topology<Topo>::cellvertices()];

private:
#ifdef USE_MPI
	MPI_Comm m_comm;
#endif // USE_MPI

	const cell_t * const m_cells;

	const idx_t m_numCells;

public:
	PartitionMetis(const cell_t* cells, unsigned int numCells) :
#ifdef USE_MPI
		m_comm(MPI_COMM_WORLD),
#endif // USE_MPI
		m_cells(cells), m_numCells(numCells)
	{ }

#ifdef USE_MPI
	void setComm(MPI_Comm comm)
	{
		m_comm = comm;
	}
#endif // USE_MPI

#ifdef USE_MPI
	/**
	 * @param partition An array of size <code>numCells</code> which
	 *  will contain the partition for each cells
         * @param vertexWeights Weight for each vertex
         * @param nWeightsPerVertex Number of weights per vertex
         * @param nodeWeights Weight for each node
	 * @param imbalance The allowed imbalance
	 */
	void partition(int* partition, int const* vertexWeights = nullptr, int nWeightsPerVertex = 1, double* nodeWeights = nullptr, double imbalance = 1.05)
	{
		int rank, procs;
		MPI_Comm_rank(m_comm, &rank);
		MPI_Comm_size(m_comm, &procs);

		idx_t* elemdist = new idx_t[procs+1];
		MPI_Allgather(const_cast<idx_t*>(&m_numCells), 1, IDX_T, elemdist, 1, IDX_T, m_comm);
		idx_t sum = 0;
		for (unsigned i = 0; i < procs; i++) {
			idx_t e = elemdist[i];
			elemdist[i] = sum;
			sum += e;
		}
		elemdist[procs] = sum;

		idx_t* eptr = new idx_t[m_numCells+1];
		idx_t* eind = new idx_t[m_numCells * internal::Topology<Topo>::cellvertices()];
		unsigned long m = 0;
		for (unsigned int i = 0; i < m_numCells; i++) {
			eptr[i] = i * internal::Topology<Topo>::cellvertices();

			for (unsigned int j = 0; j < internal::Topology<Topo>::cellvertices(); j++) {
				m = std::max(m, m_cells[i][j]);
				eind[i*internal::Topology<Topo>::cellvertices() + j] = m_cells[i][j];
			}
		}
		eptr[m_numCells] = m_numCells * internal::Topology<Topo>::cellvertices();
    
    idx_t wgtflag = 0;
    idx_t* elmwgt = nullptr;
    if (vertexWeights != nullptr) {
      wgtflag = 2;
      elmwgt = new idx_t[m_numCells];
      for (unsigned int i = 0; i < m_numCells; i++) {
        elmwgt[i] = static_cast<idx_t>(vertexWeights[i]);
      }
    }

		idx_t numflag = 0;
		idx_t ncon = nWeightsPerVertex;
		idx_t ncommonnodes = 3; // TODO adapt for hex
		idx_t nparts = procs;

		real_t* tpwgts = new real_t[nparts * ncon];
    if (nodeWeights != nullptr) {
      for (int i = 0; i < nparts; i++) {
        for (unsigned j = 0; j < ncon; ++j) {
          tpwgts[i*ncon + j] = nodeWeights[i];
        }
      }
    } else {
      for (int i = 0; i < nparts * ncon; i++) {
        tpwgts[i] = static_cast<real_t>(1.) / nparts;
      }
    }

		real_t* ubvec = new real_t[ncon];
		for (int i = 0; i < ncon; ++i) {
			ubvec[i] = imbalance;
		}
		idx_t edgecut;
		idx_t options[3] = {1, 0, METIS_RANDOM_SEED};

		idx_t* part = new idx_t[m_numCells];

		ParMETIS_V3_PartMeshKway(elemdist, eptr, eind, elmwgt, &wgtflag, &numflag,
			&ncon, &ncommonnodes, &nparts, tpwgts, ubvec, options, &edgecut, part, &m_comm);

		delete [] elemdist;
		delete [] eptr;
		delete [] eind;
		delete [] tpwgts;
		delete [] ubvec;

		for (unsigned int i = 0; i < m_numCells; i++)
			partition[i] = part[i];

		delete [] part;
	}
#endif // USE_MPI

private:
	static const int METIS_RANDOM_SEED = 42;
};

/** Convenient typedef for tetrahrdral meshes */
typedef PartitionMetis<TETRAHEDRON> TETPartitionMetis;

}

#endif // PUML_PARTITIONMETHIS_H
