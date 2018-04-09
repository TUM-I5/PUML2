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
#include <omp.h>
#include <string>

#include "utils/args.h"
#include "utils/logger.h"
#include "utils/Stopwatch.h"

#define LOCK_SIZE 512
#define CONTAINER_SIZE 8

#include "PUML.h"
#include "Downward.h"
#include "Neighbor.h"
#include "PartitionMetis.h"

#include "XdmfWriter/XdmfWriter.h"

int main(int argc, char* argv[])
{

	MPI_Init(&argc, &argv);

	int rank;
	int ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    double outer_times[5];

	Stopwatch totalTime = Stopwatch();
    totalTime.start();

	utils::Args args;
	args.addAdditionalOption("in", "the PUML mesh file");
	args.addAdditionalOption("out", "the output file");
	//args.addAdditionalOption("LOCK_SIZE", "number of cells a lock is responsible for");

	switch (args.parse(argc, argv)) {
	case utils::Args::Error:
		MPI_Abort(MPI_COMM_WORLD, -1);
		break;
	case utils::Args::Help:
		MPI_Finalize();
		return 1;
	}

	std::string infile = args.getAdditionalArgument<const char*>("in");
	std::string outfile = args.getAdditionalArgument<const char*>("out");
	//int LOCK_SIZE = std::stoi(args.getAdditionalArgument<const char*>("LOCK_SIZE"));

	PUML::TETPUML puml;


	Stopwatch readTime = Stopwatch();
	readTime.start();
	// Read the mesh
	logInfo(rank) << "Reading mesh";
	puml.open((infile + ":/connect").c_str(), (infile + ":/geometry").c_str());

	logInfo(rank) << "Reading other data";
	puml.addData((infile  + ":/group").c_str(), PUML::CELL);

    outer_times[0] = readTime.pause();
    readTime.stop();

	// Run the partitioning
	logInfo(rank) << "Creating partitions";
	
	Stopwatch partitionTime = Stopwatch();
    partitionTime.start();
	
	PUML::TETPartitionMetis metis(puml.originalCells(), puml.numOriginalCells());
	int* partition = new int[puml.numOriginalCells()];
	metis.partition(partition);

	outer_times[1] = partitionTime.pause();
    partitionTime.stop();

	// Redistribute the cells
	logInfo(rank) << "Redistributing cells";
	Stopwatch redistTime = Stopwatch();
    redistTime.start();
	puml.partition(partition);

	outer_times[2] = redistTime.pause();
    redistTime.stop();

	delete [] partition;

	Stopwatch generationTime = Stopwatch();
	generationTime.start();
	// Generate the mesh information
	logInfo(rank) << "Generating mesh information";
	puml.generateMesh();

	outer_times[3] = generationTime.pause();
    generationTime.stop();

	// Write test
	logInfo(rank) << "Writing test file";
	const std::vector<PUML::TETPUML::cell_t> &cells = puml.cells();

	const std::vector<PUML::TETPUML::vertex_t> &vertices = puml.vertices();

	xdmfwriter::XdmfWriter<xdmfwriter::TETRAHEDRON, double> writer(xdmfwriter::POSIX, outfile.c_str());

	std::vector<const char*> cellVars;
	cellVars.push_back("group");
	std::vector<const char*> vertexVars;
	writer.init(cellVars, vertexVars);

	double* coords = new double[vertices.size()*3];
	for (unsigned int i = 0; i < vertices.size(); i++) {
		memcpy(&coords[i*3], vertices[i].coordinate(), 3*sizeof(double));
	}

	unsigned int* cellPoints = new unsigned int[cells.size()*4];
	for (unsigned int i = 0; i < cells.size(); i++) {
		PUML::Downward::vertices(puml, cells[i], &cellPoints[i*4]);
	}

	writer.setMesh(cells.size(), cellPoints, vertices.size(), coords);

	writer.addTimeStep(0);

	// Write cell/vertex data
	double* cellData = new double[cells.size()];
	for (unsigned int i = 0; i < cells.size(); i++)
		cellData[i] = puml.cellData(0)[i];

	writer.writeCellData(0, cellData);

	delete [] cellData;

	writer.close();
/*
	outer_times[4] = totalTime.pause();
    totalTime.stop();


    double* inner_times_targ = new double[ranks*9];
    double* outer_times_targ = new double[ranks*5];

    MPI_Barrier(MPI_COMM_WORLD);    

    MPI_Allgather(&puml.times, 9, MPI_DOUBLE, inner_times_targ, 9, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(&outer_times, 5, MPI_DOUBLE, outer_times_targ, 5, MPI_DOUBLE, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0)
    {
    	printf("\nranks=%i,_threads_per_rank=%i,_LOCK_SIZE=%i,_container_size=%i\n\n", ranks, omp_get_max_threads(), LOCK_SIZE, CONTAINER_SIZE);
    }

    if(rank == 0)
    {
	    double average[9] = {0,0,0,0,0,0,0,0,0};
	    double fastest[9] = {99999999,99999999,99999999,99999999,99999999,99999999,99999999,99999999,99999999};
	    double slowest[9] = {0,0,0,0,0,0,0,0,0};
	    std::string measurements[9] = "";

	    for(int j = 0; j < 9; j++)
	    {
		    for(int i = j; i < ranks*9; i = i+9)
		    {
		        average[j] += inner_times_targ[i];
		        if(inner_times_targ[i] < fastest[j])
		            fastest[j] = inner_times_targ[i];
		        if(inner_times_targ[i] > slowest[j])
		            slowest[j] = inner_times_targ[i];
		    }
		    average[j] = average[j] / ranks;
		    measurements[j] = std::to_string(fastest[j]);
		    if(ranks > 1)
		    {
		    	measurements[j] += " ";
		    	measurements[j] += std::to_string(average[j]);
		    	measurements[j] += " ";
		    	measurements[j] += std::to_string(slowest[j]);
		    }

		}
		
	    printf("generate_mesh()_measurements:\n");
	    printf("Manage_boundaries_and_redistribute %s\n", measurements[0].c_str());
	    printf("Insert_faces %s\n", measurements[1].c_str());
	    printf("Number_faces_and_save %s\n", measurements[2].c_str());
	    printf("Insert_edges_and_save_vertex_upward %s\n", measurements[3].c_str());
	    printf("Number_edges_and_save_edge_upward %s\n", measurements[4].c_str());
	    printf("Update_vertex_upward %s\n", measurements[5].c_str());
	    printf("Adjacency_time %s\n", measurements[6].c_str());
	    printf("Global_ID_time %s\n", measurements[7].c_str());
	    printf("Generate_mesh_time %s\n", measurements[8].c_str());
	}

	if(rank == 0)
    {
	    double average[5] = {0,0,0,0,0};
	    double fastest[5] = {99999999,99999999,99999999,99999999,99999999};
	    double slowest[5] = {0,0,0,0,0};
	    std::string measurements[5] = "";

	    for(int j = 0; j < 5; j++)
	    {
		    for(int i = j; i < ranks*5; i = i+5)
		    {
		        average[j] += outer_times_targ[i];
		        if(outer_times_targ[i] < fastest[j])
		            fastest[j] = outer_times_targ[i];
		        if(outer_times_targ[i] > slowest[j])
		            slowest[j] = outer_times_targ[i];
		    }
		    average[j] = average[j] / ranks;
		    measurements[j] = std::to_string(fastest[j]);
		    if(ranks > 1)
		    {
		    	measurements[j] += " ";
		    	measurements[j] += std::to_string(average[j]);
		    	measurements[j] += " ";
		    	measurements[j] += std::to_string(slowest[j]);
		    }

		}
		
	    printf("\nPUML_measurements:\n");
	    printf("Input_time %s\n", measurements[0].c_str());
	    printf("Partition_time %s\n", measurements[1].c_str());
	    printf("Redistributing_time %s\n", measurements[2].c_str());
	    printf("Generate_time %s\n", measurements[3].c_str());
	    printf("Total_time %s\n", measurements[4].c_str());
	}
*/
	MPI_Finalize();

	return 0;
}