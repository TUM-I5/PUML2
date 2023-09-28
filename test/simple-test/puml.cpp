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

#include <mpi.h>

#include "utils/args.h"
#include "utils/logger.h"

#include "PUML.h"
#include "Downward.h"
#include "Neighbor.h"
#include "Partition.h"
#include "PartitionGraph.h"

#include <sstream>
#include <vector>
#include <type_traits>

template<typename T>
void printElement(std::stringstream& stream, const T& data) {
	if constexpr (std::is_array_v<T>) {
		for (std::size_t i = 0; i < std::extent_v<T, 0>; ++i) {
			if (i > 0) {
				stream << ",";
			}
			printElement(stream, data[i]);
		}
	}
	else {
		stream << data;
	}
}

/*template<>
static void printElement<PUML::TETPUML::cell_t>(std::stringstream& stream, const PUML::TETPUML::cell_t& data) {
	PUML::Downward::vertices();
}*/

template<>
void printElement<PUML::TETPUML::vertex_t>(std::stringstream& stream, const PUML::TETPUML::vertex_t& data) {
	const auto& cref = data.coordinate();
	double coords[3] = {cref[0], cref[1], cref[2]};
	printElement(stream, coords);
}

template<typename T>
static void printArray(const T* data, std::size_t size) {
	int rank;
	int rsize;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &rsize);

	unsigned long long countSoFar = 0;

	if (rank > 0) {
		MPI_Recv(&countSoFar, 1, MPI_UNSIGNED_LONG_LONG, rank - 1, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	logInfo(0) << "Rank" << rank;

	std::stringstream stream;
	for (std::size_t i = 0; i < size; ++i) {
		stream << countSoFar << ": ";
		printElement(stream, data[i]);
		logInfo(0) << stream.str().c_str();
		stream.str("");
		++countSoFar;
	}

	if (rank < rsize - 1) {
		MPI_Send(&countSoFar, 1, MPI_UNSIGNED_LONG_LONG, rank + 1, 15, MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

template<typename T>
static void printArray(const std::vector<T>& data) {
	printArray(data.data(), data.size());
}

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	utils::Args args;
	args.addAdditionalOption("in", "the PUML mesh file");

	switch (args.parse(argc, argv)) {
	case utils::Args::Error:
		MPI_Abort(MPI_COMM_WORLD, -1);
		break;
	case utils::Args::Help:
		MPI_Finalize();
		return 1;
	}

	PUML::TETPUML puml;

	std::string infile = args.getAdditionalArgument<const char*>("in");

	// Read the mesh
	logInfo(rank) << "Reading mesh";
	puml.open((infile + ":/connect").c_str(), (infile + ":/geometry").c_str());

	logInfo(rank) << "Reading other data (i.e. groups, boundaries)";
	puml.addData((infile  + ":/group").c_str(), PUML::CELL);
	puml.addData((infile  + ":/boundary").c_str(), PUML::CELL);

	std::vector<unsigned long long> test(puml.numOriginalCells(), 0x5555555555555555ULL);
	puml.addData(test.data(), puml.numOriginalCells(), PUML::CELL);

	// Generate the mesh information
	logInfo(rank) << "Generating mesh information";
	puml.generateMesh();

	// Write test
	logInfo(rank) << "Printing test information";
	const std::vector<PUML::TETPUML::cell_t> &cells = puml.cells();

	const std::vector<PUML::TETPUML::vertex_t> &vertices = puml.vertices();
	printArray(vertices);

	const int* groups = reinterpret_cast<const int*>(puml.cellData(0));
	printArray(groups, cells.size());
	const unsigned long long* testt = reinterpret_cast<const unsigned long long*>(puml.cellData(2));
	printArray(testt, cells.size());

	logInfo(rank) << "Done";

	MPI_Finalize();

	return 0;
}