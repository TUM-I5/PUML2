/**
 * @file
 *  This file is part of PUML
 *
 *  For conditions of distribution and use, please see the copyright
 *  notice in the file 'COPYING' at the root directory of this package
 *  and the copyright notice at https://github.com/TUM-I5/PUMGen
 *
 * @copyright 2019 Technische Universitaet Muenchen
 * @author Sebastian Rettenberger <sebastian.rettenberger@tum.de>
 * @author David Schneller <david.schneller@tum.de>
 */

#include <mpi.h>

#include <sys/time.h>
#include <sys/resource.h>

#include "utils/args.h"
#include "utils/logger.h"
#include "nlohmann/json.hpp"

#include "PUML.h"
#include "Downward.h"
#include "Neighbor.h"
#include "PartitionGraph.h"

#include <random>
#include <iostream>
#include <fstream>
#include <cmath>
#include <functional>
#include <string>

#include "Partition.h"

std::vector<nlohmann::json> argparse(int argc, char* argv[]) {
	std::vector<nlohmann::json> settings {
		nlohmann::json()
	};
	for (int i = 1; i < argc; ++i) {
		nlohmann::json js;
		std::string carg(argv[i]);
		if (carg.find('=') != std::string::npos) {
			size_t idx = carg.find('=');
			std::string name = carg.substr(0, idx);
			std::string val = carg.substr(idx + 1);
			js[name] = val;
		}
		else {
			std::ifstream settings_file(carg);
			settings_file >> js;
		}
		
		std::vector<nlohmann::json> jss;
		if (js.is_array()) {
			jss = std::vector<nlohmann::json>(js.begin(), js.end());
		}
		else {
			jss = std::vector<nlohmann::json> { js };
		}

		std::vector<nlohmann::json> nsettings(jss.size() * settings.size());
		for (int j = 0, l = 0; j < jss.size(); ++j) {
			for (int k = 0; k < settings.size(); ++k, ++l) {
				nlohmann::json copy = settings[k];
				copy.merge_patch(jss[j]);
				nsettings[l] = copy;
			}
		}
		settings = nsettings;
	}
	return settings;
}

int main(int argc, char* argv[])
{
	int p;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &p);

	MPI_Comm comm = MPI_COMM_WORLD;
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	if (argc == 1) {
		logWarning(rank) << "No settings given. They can be set as command line arguments. "
		    "An argument can have two forms, either a key-value pair in the form \"key=value\" (note the '=', and no spaces around it), "
			"or it can be a file name which contains a JSON file (note in particular: you may mix files and key-value pairs, and you may supply multiple files). "
			"Later arguments override earlier arguments.";
	}

	auto settings_list = argparse(argc, argv);

	
	logInfo(rank) << "PUML Partition Tester --- Starting";
	logInfo(rank) << settings_list.size() << "tests";

	PUML::TETPUML* puml = nullptr;

	std::string lastmesh = "";
	std::string lastmatmesh = "";
	std::string lastmat = "";

	for (int i = 0; i < settings_list.size(); ++i) {
		auto settings = settings_list[i];
		logInfo(rank) << "Test" << (i+1) << "of" << settings_list.size();
		logInfo(rank) << "Using configuration: " << settings;
		if (settings.is_null()) {
			logInfo(rank) << "Aborting...";
			break;
		}

		if (puml == nullptr || settings["meshfile"] != lastmesh) {
			delete puml;
			puml = new PUML::TETPUML();
			std::string infile = settings["meshfile"];
			logInfo(rank) << "Reading mesh" << infile;
			puml->open((infile + ":/connect").c_str(), (infile + ":/geometry").c_str());

			logInfo(rank) << "Reading other data";
			puml->addData((infile + ":/group").c_str(), PUML::CELL);
			puml->addData((infile + ":/boundary").c_str(), PUML::CELL);

			logInfo(rank) << "Generating mesh information";
			puml->generateMesh();

			lastmesh = settings["meshfile"];
		}

		std::string partitioner_name = settings["name"];
		auto partition_invoker = PUML::TETPartition::get_partitioner(partitioner_name);
		if (partition_invoker == nullptr) {
			logInfo(rank) << "Partitioner name not recognized. Aborting...";
			MPI_Finalize();
			return 1;
		}

		logInfo(rank) << "Creating partitions using partitioner" << partitioner_name;
		std::vector<int> partition(puml->numOriginalCells());
		PUML::TETPartitionGraph graph(*puml);

		std::string vertexweightsetting = settings["vw"];
		std::string edgeweightsetting = settings["ew"];
		std::string nodeweightsetting = settings["nw"];

		int * vertexweights = nullptr;
		int * edgeweights = nullptr;
		double * nodeweights = nullptr;
		if (vertexweightsetting == "none") {
			// blank
		}
		else if (vertexweightsetting == "uniform") {
			vertexweights = new int[puml->cells().size()];
			for (int i = 0; i < puml->numOriginalCells(); ++i) {
				vertexweights[i] = 1;
			}
		}
		else if (vertexweightsetting == "random") {
			vertexweights = new int[puml->cells().size()];
			std::default_random_engine gen(123);
			std::uniform_int_distribution<int> distribution(1,1000);
			for (int i = 0; i < puml->numOriginalCells(); ++i) {
				vertexweights[i] = distribution(gen);
			}
		}
		else {
			logInfo(rank) << "Vertex weight setting not recognized. Aborting...";
			MPI_Finalize();
			return 1;
		}

		if (edgeweightsetting == "none") {
			// blank
		}
		else if (edgeweightsetting == "uniform") {
			edgeweights = new int[graph.local_edge_count()];
			for (int i = 0; i < graph.local_edge_count(); ++i) {
				edgeweights[i] = 1;
			}
		}
		else if (edgeweightsetting == "random") {
			edgeweights = new int[graph.local_edge_count()];
			std::default_random_engine gen(123);
			std::uniform_int_distribution<int> distribution(1,1000);
			for (int i = 0; i < graph.local_edge_count(); ++i) {
				edgeweights[i] = distribution(gen);
			}
		}
		else {
			logInfo(rank) << "Edge weight setting not recognized. Aborting...";
			MPI_Finalize();
			return 1;
		}

		int nparts = settings["nparts"];

		if (nodeweightsetting == "none") {
			// empty
		}
		else if (nodeweightsetting == "uniform") {
			nodeweights = new double[nparts];
			for (int i = 0; i < nparts; ++i) {
				nodeweights[i] = 1. / nparts;
			}
		}
		else if (nodeweightsetting == "random") {
			nodeweights = new double[nparts];
			std::default_random_engine gen(123);
			std::uniform_int_distribution<int> distribution(1,1000);
			double sum = 0;
			for (int i = 0; i < nparts; ++i) {
				nodeweights[i] = distribution(gen);
				sum += nodeweights[i];
			}
			for (int i = 0; i < nparts; ++i) {
				nodeweights[i] /= sum;
			}
		}
		else if (nodeweightsetting == "harmonic") {
			nodeweights = new double[nparts];
			double sum = 0;
			for (int i = 0; i < nparts; ++i) {
				sum += 1. / (i + 1);
			}
			for (int i = 0; i < nparts; ++i) {
				nodeweights[i] = 1. / (sum * (i + 1));
			}
		}

		graph.set_vertex_weights(vertexweights, 1);
		graph.set_edge_weights(edgeweights);
		double imbalance = settings["imbalance"];
		int seed = settings["seed"];

		PUML::PartitionTarget target;
		target.set_imbalance(imbalance);
		if (nodeweights == nullptr) {
			target.set_vertex_weights_uniform(nparts);
		}
		else {
			target.set_vertex_weights(nparts, nodeweights);
		}

		logDebug() << "Initial stats" << rank << graph.local_vertex_count() << graph.local_edge_count();

		logInfo(rank) << "Starting the actual partitioning.";
		struct rusage startm;
		getrusage(RUSAGE_SELF, &startm);
		double startt = MPI_Wtime();
		partition_invoker->partition(partition.data(), graph, target, seed);
		MPI_Barrier(comm);
		double endt = MPI_Wtime();
		struct rusage endm;
		getrusage(RUSAGE_SELF, &endm);
		logInfo(rank) << "Actual partitioning done. Now saving stats";

		// counts weighted and unweighted sizes
		std::vector<int> partition_size(nparts*4);
		std::vector<int> partition_size_local(nparts*4);
		for (int i = 0; i < puml->cells().size(); ++i) {
			++partition_size_local[partition[i]];
		}
		if (vertexweights == nullptr) {
			for (int i = 0; i < nparts; ++i) {
				partition_size_local[i + nparts] = partition_size_local[i];
			}
		}
		else {
			for (int i = 0; i < puml->cells().size(); ++i) {
				partition_size_local[partition[i] + nparts] += vertexweights[i];
			}
		}

		std::vector<int> detail_partition(nparts*nparts);
		std::vector<int> local_detail_partition(nparts*nparts);

		graph.forall_local_edges<int>(
			partition,
			[&](int fid, int lid, const int& g, const int& l, int id) {
				if (g != l) {
					++partition_size_local[g + nparts * 2];
					if (edgeweights == nullptr) {
						++partition_size_local[g + nparts * 3];
						++local_detail_partition[g * nparts + l];
					}
					else {
						partition_size_local[g + nparts * 3] += edgeweights[id];
						local_detail_partition[g * nparts + l] += edgeweights[id];
					}
				}
			}
		);

		MPI_Reduce(partition_size_local.data(), partition_size.data(), partition_size_local.size(), MPI_INT, MPI_SUM, 0, comm);
		MPI_Reduce(local_detail_partition.data(), detail_partition.data(), local_detail_partition.size(), MPI_INT, MPI_SUM, 0, comm);

		long peakmemstart = startm.ru_maxrss;
		long peakmemend = endm.ru_maxrss;

		long peakmemlocal[] = {peakmemstart, peakmemend};
		long peakmem[2];
		long avgpeakmem[2];
		MPI_Reduce(peakmemlocal, peakmem, 2, MPI_LONG, MPI_MAX, 0, comm);
		MPI_Reduce(peakmemlocal, avgpeakmem, 2, MPI_LONG, MPI_SUM, 0, comm);

		double realavgpeakmem[2];
		realavgpeakmem[0] = (double)avgpeakmem[0] / size;
		realavgpeakmem[1] = (double)avgpeakmem[1] / size;

		std::vector<long> peakmemrankstart(size);
		std::vector<long> peakmemrankend(size);

		MPI_Gather(&startm.ru_maxrss, 1, MPI_LONG, peakmemrankstart.data(), 1, MPI_LONG, 0, comm);
		MPI_Gather(&endm.ru_maxrss, 1, MPI_LONG, peakmemrankend.data(), 1, MPI_LONG, 0, comm);

		if (rank == 0) {
			std::vector<nlohmann::json> node_data(nparts);
			size_t ec = 0;
			size_t ecw = 0;
			size_t ps = 0;
			size_t psw = 0;
			size_t pss = 0;
			size_t psws = 0;
			for (int i = 0; i < nparts; ++i) {
				node_data[i]["size"] = partition_size[i];
				node_data[i]["size_weighted"] = partition_size[i + nparts];
				node_data[i]["cut"] = partition_size[i + nparts * 2];
				node_data[i]["cut_weighted"] = partition_size[i + nparts * 3];

				std::vector<int> detail(detail_partition.begin() + nparts * i, detail_partition.begin() + nparts * (i+1));
				node_data[i]["cut_weighted_detail"] = detail;

				ec += partition_size[i + nparts*2];
				ecw += partition_size[i + nparts*3];
				ps = std::max(ps, (size_t)partition_size[i]);
				psw = std::max(psw, (size_t)partition_size[i + nparts]);
				pss += partition_size[i];
				psws += partition_size[i + nparts];
			}
			ec /= 2; ecw /= 2;
			double mi = (double)ps / ((double)pss / (double)nparts);
			double miw = (double)psw / ((double)psws / (double)nparts);
			logInfo(rank) << ec << ecw << mi << miw << peakmem[0] << peakmem[1] << (endt - startt);

			nlohmann::json output;

			output["cut"] = ec;
			output["cut_weighted"] = ecw;
			output["imbalance"] = mi;
			output["imbalance_weighted"] = miw;
			output["total_vertices"] = graph.global_vertex_count();
			output["total_edges"] = graph.global_edge_count();
			output["mpi_size"] = size;
			output["partition_stats"] = node_data;
			output["time"] = endt-startt;
			output["configuration"] = settings;

			output["peakmembeforemax"] = peakmem[0];
			output["peakmemaftermax"] = peakmem[1];
			output["peakmembeforeavg"] = realavgpeakmem[0];
			output["peakmemafteravg"] = realavgpeakmem[1];

			std::vector<nlohmann::json> rankdata(size);
			for (int i = 0; i < size; ++i) {
				rankdata[i]["peakmembefore"] = peakmemrankstart[i];
				rankdata[i]["peakmemafter"] = peakmemrankend[i];
			}

			output["rank_stats"] = rankdata;

			std::string odir = settings["output"];
			size_t hash = std::hash<nlohmann::json>()(output);
			std::ofstream output_file(odir + "/" + std::to_string(hash) + ".json");
			output_file << output;
			if (settings_list.size() <= 4) {
				logInfo(rank) << "Results: " << output;
			}
		}

		if (vertexweights != nullptr) {
			delete[] vertexweights;
		}
		if (edgeweights != nullptr) {
			delete[] edgeweights;
		}
		if (nodeweights != nullptr) {
			delete[] nodeweights;
		}

		logInfo(rank) << "Done";
	}
	MPI_Finalize();

	return 0;
}