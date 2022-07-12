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

#ifdef USE_MPI
	int m_rank = 0;
	int m_nparts = 0;
#endif

	// Better the use the same convention as METIS, even though we have the prefix m_
	idx_t m_numflag = 0;	// Since we are a C++ library, numflag for Metis will be always 0
	idx_t m_ncommonnodes = 3; // TODO adapt for hex, but until then we can use constexpr for these

	std::vector<idx_t> m_vtxdist;
	std::vector<idx_t> m_xadj;
	std::vector<idx_t> m_adjncy;

public:
	PartitionMetis(const cell_t* cells, unsigned int numCells) :
#ifdef USE_MPI
		m_comm(MPI_COMM_WORLD),
#endif // USE_MPI
		m_cells(cells), m_numCells(numCells)
	{ 
#ifdef USE_MPI
		MPI_Comm_rank(m_comm, &m_rank);
		MPI_Comm_size(m_comm, &m_nparts);
#endif
	}

#ifdef USE_MPI
	void setComm(MPI_Comm comm)
	{
		m_comm = comm;
	}
#endif // USE_MPI

#ifdef USE_MPI
  void generateGraphFromMesh() {
	assert((m_vtxdist.empty() && m_xadj.empty() && m_adjncy.empty()) || (!m_vtxdist.empty() && !m_xadj.empty() && !m_adjncy.empty()));

	if (m_vtxdist.empty() && m_xadj.empty() && m_adjncy.empty()) {
	  std::vector<idx_t> elemdist;
	  elemdist.resize(m_nparts + 1);
	  std::fill(elemdist.begin(), elemdist.end(), 0);

	  MPI_Allgather(const_cast<idx_t*>(&m_numCells), 1, IDX_T, elemdist.data(), 1, IDX_T, m_comm);

	  idx_t sum = 0;
	  for (int i = 0; i < m_nparts; i++) {
		idx_t e = elemdist[i];
		elemdist[i] = sum;
		sum += e;
	  }
	  elemdist[m_nparts] = sum;

	  std::vector<idx_t> eptr;
	  eptr.resize(m_numCells + 1);
	  std::fill(eptr.begin(), eptr.end(), 0);

	  std::vector<idx_t> eind;
	  eind.resize(m_numCells * internal::Topology<Topo>::cellvertices());
	  std::fill(eind.begin(), eind.end(), 0);

	  unsigned long m = 0;

	  for (idx_t i = 0; i < m_numCells; i++) {
		eptr[i] = i * internal::Topology<Topo>::cellvertices();

		for (unsigned int j = 0; j < internal::Topology<Topo>::cellvertices(); j++) {
		  m = std::max(m, m_cells[i][j]);
		  eind[i * internal::Topology<Topo>::cellvertices() + j] = m_cells[i][j];
		}
	  }

	  eptr[m_numCells] = m_numCells * internal::Topology<Topo>::cellvertices();

	  idx_t* metis_xadj;
	  idx_t* metis_adjncy;

	  ParMETIS_V3_Mesh2Dual(elemdist.data(), eptr.data(), eind.data(), &m_numflag, &m_ncommonnodes, &metis_xadj,
							&metis_adjncy, &m_comm);

	  m_vtxdist = std::move(elemdist);

	  //  the size of xadj is the
	  //  - vtxdist[proc] + vtxdist[proc+1]
	  //  because proc has the index proc to proc +1 elements

	  assert(m_vtxdist.size() == static_cast<size_t>(m_nparts + 1));
	  // the first element is always 0 and on top of that we have n nodes
	  size_t numElements = m_vtxdist[m_rank + 1] - m_vtxdist[m_rank] + 1;
	  m_xadj.reserve(numElements);
	  std::copy(metis_xadj, metis_xadj + numElements, std::back_inserter(m_xadj));

	  // last element of xadj will be the size of adjncy
	  size_t adjncySize = m_xadj[numElements - 1];
	  m_adjncy.reserve(adjncySize);
	  std::copy(metis_adjncy, metis_adjncy + adjncySize, std::back_inserter(m_adjncy));

	  METIS_Free(metis_xadj);
	  METIS_Free(metis_adjncy);
	}
  }
#endif

#ifdef USE_MPI
  std::tuple<const std::vector<idx_t>&, const std::vector<idx_t>&, const std::vector<idx_t>&> getGraph() {
	if (m_xadj.empty() && m_adjncy.empty()) {
	  generateGraphFromMesh();
	}

	return {m_vtxdist, m_xadj, m_adjncy};
  }
#endif

#ifdef USE_MPI
	enum Status {
		Ok,
		Error
	};

	/**
	 * @param partition An array of size <code>numCells</code> which
	 *  will contain the partition for each cells
	 * @param vertexWeights Weight for each vertex
	 * @param nWeightsPerVertex Number of weights per vertex
	 * @param nodeWeights Weight for each node
	 * @param imbalance The allowed imbalance for each constrain
	 */
	Status partition(int* partition,
		const int* vertexWeights = nullptr,
		const double* imbalances = nullptr,
		int nWeightsPerVertex = 1,
		const double* nodeWeights = nullptr,
		const int* edgeWeights = nullptr,
		size_t edgeCount = 0) 
	{
		generateGraphFromMesh();

		idx_t wgtflag = 0;

		// set the flag
		if (nodeWeights == nullptr && edgeWeights == nullptr) {
			wgtflag = 0;
		} else if (nodeWeights != nullptr && edgeWeights != nullptr) {
			wgtflag = 3;
		} else if (nodeWeights == nullptr && edgeWeights != nullptr) {
			wgtflag = 1;
		} else {
			wgtflag = 2;
		}

		idx_t ncon = nWeightsPerVertex;
		idx_t* elmwgt = nullptr;
		
		if (vertexWeights != nullptr) {
			elmwgt = new idx_t[m_numCells * ncon];
			for (idx_t cell = 0; cell < m_numCells; ++cell) {
				for (idx_t j = 0; j < ncon; ++j) {
					elmwgt[ncon * cell + j] = static_cast<idx_t>(vertexWeights[ncon * cell + j]);
				}
			}
		}

		idx_t* edgewgt = nullptr;
		if (edgeWeights != nullptr) {
			assert(edgeCount != 0);
			edgewgt = new idx_t[edgeCount];
			for (size_t i = 0; i < edgeCount; ++i) {
				edgewgt[i] = static_cast<idx_t>(edgeWeights[i]);
			}
		}

		real_t* tpwgts = new real_t[m_nparts* ncon];
		if (nodeWeights != nullptr) {
			for (idx_t i = 0; i < m_nparts; i++) {
				for (idx_t j = 0; j < ncon; ++j) {
					tpwgts[i * ncon + j] = nodeWeights[i];
				}
			}
		} else {
			for (idx_t i = 0; i < m_nparts * ncon; i++) {
				tpwgts[i] = static_cast<real_t>(1.) / m_nparts;
			}
		}

		real_t* ubvec = new real_t[ncon];
		for (idx_t i = 0; i < ncon; ++i) {
			ubvec[i] = imbalances[i];
		}

		idx_t edgecut;
		idx_t options[3] = {1, 1, METIS_RANDOM_SEED};

		idx_t* part = new idx_t[m_numCells];

		assert(m_xadj.size() == static_cast<size_t>(m_vtxdist[m_rank + 1] - m_vtxdist[m_rank] + 1));
		assert(m_adjncy.size() == static_cast<size_t>(m_xadj.back()));

		auto metisResult = ParMETIS_V3_PartKway(m_vtxdist.data(), m_xadj.data(), m_adjncy.data(), elmwgt, edgewgt, &wgtflag,
												&m_numflag, &ncon, &m_nparts, tpwgts, ubvec, options, &edgecut, part, &m_comm);
												
		delete[] tpwgts;
		delete[] ubvec;
		delete[] elmwgt;
		delete[] edgewgt;

		for (idx_t i = 0; i < m_numCells; i++){
			partition[i] = static_cast<int>(part[i]);
		}
		
		delete[] part;

		return (metisResult == METIS_OK) ? Status::Ok : Status::Error;
	}
#endif // USE_MPI

private:
	static const int METIS_RANDOM_SEED = 42;
};

/** Convenient typedef for tetrahrdral meshes */
typedef PartitionMetis<TETRAHEDRON> TETPartitionMetis;

}

#endif // PUML_PARTITIONMETHIS_H
