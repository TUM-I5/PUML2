// SPDX-FileCopyrightText: 2019-2023 Technical University of Munich
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
  std::vector<nlohmann::json> settings{nlohmann::json()};
  for (int i = 1; i < argc; ++i) {
    nlohmann::json js;
    std::string carg(argv[i]);
    if (carg.find('=') != std::string::npos) {
      size_t idx = carg.find('=');
      std::string name = carg.substr(0, idx);
      std::string val = carg.substr(idx + 1);

      // a quick&dirty argument type evaluator (it should actually use some trimming as well, but
      // that's TODO, in case it should ever be needed)
      if ((*val.begin()) == '\"' && (*val.rbegin()) == '\"') {
        // string
        js[name] = val.substr(1, val.size() - 2);
      } else if (val == "true") {
        // true bool
        js[name] = true;
      } else if (val == "false") {
        // false bool
        js[name] = false;
      } else {
        // number (represented by a double)
        js[name] = std::stod(val);
      }
    } else {
      std::ifstream settings_file(carg);
      settings_file >> js;
    }

    std::vector<nlohmann::json> jss;
    if (js.is_array()) {
      jss = std::vector<nlohmann::json>(js.begin(), js.end());
    } else {
      jss = std::vector<nlohmann::json>{js};
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

template <typename T>
T readSetting(const nlohmann::json& settings, const std::string& arg) {
  if (settings.find(arg) != settings.end()) {
    try {
      return settings[arg];
    } catch (const std::exception& e) {
      logError() << "Error while reading argument" << arg << " ... The error was" << e.what();
      return T(); // TODO: unreachable hint
    }
  } else {
    logError() << "Argument" << arg << "not found in the settings list.";
    return T(); // TODO: unreachable hint
  }
}

int main(int argc, char* argv[]) {
  int p;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &p);

  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (argc == 1) {
    logWarning(rank)
        << "No settings given. They can be set as command line arguments. "
           "An argument can have two forms, either a key-value pair in the form \"key=value\" "
           "(note the '=', and no spaces around it), "
           "or it can be a file name which contains a JSON file (note in particular: you may mix "
           "files and key-value pairs, and you may supply multiple files). "
           "Later arguments override earlier arguments. If a file contains an array of "
           "configurations, each configuration is tested."
           "Any following argument is then applied to all configurations. If again an array of "
           "configurations is given, the cross product between the two arrays is used.";
  }

  auto settingsList = argparse(argc, argv);

  logInfo(rank) << "PUML Partition Tester --- Starting";
  logInfo(rank) << settingsList.size() << "tests";

  PUML::TETPUML* puml = nullptr;

  std::string lastmesh = "";
  std::string lastmatmesh = "";
  std::string lastmat = "";

  for (int i = 0; i < settingsList.size(); ++i) {
    auto settings = settingsList[i];
    logInfo(rank) << "Test" << (i + 1) << "of" << settingsList.size();
    logInfo(rank) << "Using configuration: " << settings;
    if (settings.is_null()) {
      logInfo(rank) << "Aborting...";
      break;
    }

    if (puml == nullptr || readSetting<std::string>(settings, "meshfile") != lastmesh) {
      delete puml;
      puml = new PUML::TETPUML();
      std::string infile = readSetting<std::string>(settings, "meshfile");
      logInfo(rank) << "Reading mesh" << infile;
      puml->open((infile + ":/connect").c_str(), (infile + ":/geometry").c_str());

      logInfo(rank) << "Reading other data";
      puml->addData((infile + ":/group").c_str(), PUML::CELL);
      puml->addData((infile + ":/boundary").c_str(), PUML::CELL);

      logInfo(rank) << "Generating mesh information";
      puml->generateMesh();

      lastmesh = readSetting<std::string>(settings, "meshfile");
    }

    std::string partitionerName = readSetting<std::string>(settings, "name");
    PUML::PartitionerType partitionerType = PUML::PartitionerType::None;

    if (partitionerName == "none") {
      partitionerType = PUML::PartitionerType::None;
    } else if (partitionerName == "parmetis") {
      partitionerType = PUML::PartitionerType::Parmetis;
    } else if (partitionerName == "parmetis-geo") {
      partitionerType = PUML::PartitionerType::ParmetisGeometric;
    } else if (partitionerName == "ptscotch") {
      partitionerType = PUML::PartitionerType::PtScotch;
    } else if (partitionerName == "ptscotch-b") {
      partitionerType = PUML::PartitionerType::PtScotchBalance;
    } else if (partitionerName == "ptscotch-q") {
      partitionerType = PUML::PartitionerType::PtScotchQuality;
    } else if (partitionerName == "ptscotch-bq") {
      partitionerType = PUML::PartitionerType::PtScotchBalanceQuality;
    } else if (partitionerName == "ptscotch-s") {
      partitionerType = PUML::PartitionerType::PtScotchSpeed;
    } else if (partitionerName == "ptscotch-sb") {
      partitionerType = PUML::PartitionerType::PtScotchBalanceSpeed;
    } else if (partitionerName == "parhip-ultrafast") {
      partitionerType = PUML::PartitionerType::ParHIPUltrafastMesh;
    } else if (partitionerName == "parhip-fast") {
      partitionerType = PUML::PartitionerType::ParHIPFastMesh;
    } else if (partitionerName == "parhip-eco") {
      partitionerType = PUML::PartitionerType::ParHIPEcoMesh;
    } else if (partitionerName == "parhip-ultrafastsocial") {
      partitionerType = PUML::PartitionerType::ParHIPUltrafastSocial;
    } else if (partitionerName == "parhip-fastsocial") {
      partitionerType = PUML::PartitionerType::ParHIPFastSocial;
    } else if (partitionerName == "parhip-ecosocial") {
      partitionerType = PUML::PartitionerType::ParHIPEcoSocial;
    } else {
      logInfo(rank) << "Partitioner name not recognized. Aborting...";
      MPI_Finalize();
      return 1;
    }

    auto partitionInvoker = PUML::TETPartition::getPartitioner(partitionerType);
    if (partitionInvoker == nullptr) {
      logInfo(rank) << "Partitioner name not recognized. Aborting...";
      MPI_Finalize();
      return 1;
    }

    logInfo(rank) << "Creating partitions using partitioner" << partitionerName;
    std::vector<int> partition(puml->numOriginalCells());
    PUML::TETPartitionGraph graph(*puml);

    std::string vertexweightsetting = readSetting<std::string>(settings, "vw");
    std::string edgeweightsetting = readSetting<std::string>(settings, "ew");
    std::string nodeweightsetting = readSetting<std::string>(settings, "nw");

    int* vertexweights = nullptr;
    int* edgeweights = nullptr;
    double* nodeweights = nullptr;
    if (vertexweightsetting == "none") {
      // blank
    } else if (vertexweightsetting == "uniform") {
      vertexweights = new int[puml->cells().size()];
      for (int i = 0; i < puml->numOriginalCells(); ++i) {
        vertexweights[i] = 1;
      }
    } else if (vertexweightsetting == "random") {
      vertexweights = new int[puml->cells().size()];
      std::default_random_engine gen(123);
      std::uniform_int_distribution<int> distribution(1, 1000);
      for (int i = 0; i < puml->numOriginalCells(); ++i) {
        vertexweights[i] = distribution(gen);
      }
    } else if (vertexweightsetting == "increasing") {
      vertexweights = new int[puml->cells().size()];
      for (int i = 0; i < puml->numOriginalCells(); ++i) {
        vertexweights[i] = i + 1;
      }
    } else {
      logInfo(rank) << "Vertex weight setting not recognized. Aborting...";
      MPI_Finalize();
      return 1;
    }

    if (edgeweightsetting == "none") {
      // blank
    } else if (edgeweightsetting == "uniform") {
      edgeweights = new int[graph.localEdgeCount()];
      for (int i = 0; i < graph.localEdgeCount(); ++i) {
        edgeweights[i] = 1;
      }
    } else if (edgeweightsetting == "random") {
      edgeweights = new int[graph.localEdgeCount()];
      std::default_random_engine gen(123);
      std::uniform_int_distribution<int> distribution(1, 1000);
      for (int i = 0; i < graph.localEdgeCount(); ++i) {
        edgeweights[i] = distribution(gen);
      }
    } else if (edgeweightsetting == "increasing") {
      edgeweights = new int[graph.localEdgeCount()];
      for (int i = 0; i < graph.localEdgeCount(); ++i) {
        edgeweights[i] = i + 1;
      }
    } else {
      logInfo(rank) << "Edge weight setting not recognized. Aborting...";
      MPI_Finalize();
      return 1;
    }

    int nparts = readSetting<int>(settings, "nparts");

    if (nodeweightsetting == "none") {
      // empty
    } else if (nodeweightsetting == "uniform") {
      nodeweights = new double[nparts];
      for (int i = 0; i < nparts; ++i) {
        nodeweights[i] = 1. / nparts;
      }
    } else if (nodeweightsetting == "random") {
      nodeweights = new double[nparts];
      std::default_random_engine gen(123);
      std::uniform_int_distribution<int> distribution(1, 1000);
      double sum = 0;
      for (int i = 0; i < nparts; ++i) {
        nodeweights[i] = distribution(gen);
        sum += nodeweights[i];
      }
      for (int i = 0; i < nparts; ++i) {
        nodeweights[i] /= sum;
      }
    } else if (nodeweightsetting == "harmonic") {
      nodeweights = new double[nparts];
      double sum = 0;
      for (int i = 0; i < nparts; ++i) {
        sum += 1. / (i + 1);
      }
      for (int i = 0; i < nparts; ++i) {
        nodeweights[i] = 1. / (sum * (i + 1));
      }
    }

    graph.setVertexWeights(vertexweights, 1);
    graph.setEdgeWeights(edgeweights);
    double imbalance = readSetting<double>(settings, "imbalance");
    int seed = readSetting<int>(settings, "seed");

    PUML::PartitionTarget target;
    target.setImbalance(imbalance);
    if (nodeweights == nullptr) {
      target.setVertexWeightsUniform(nparts);
    } else {
      target.setVertexWeights(nparts, nodeweights);
    }

    logDebug() << "Initial stats" << rank << graph.localVertexCount() << graph.localEdgeCount();

    logInfo(rank) << "Starting the actual partitioning.";
    struct rusage memoryStatsBegin;
    getrusage(RUSAGE_SELF, &memoryStatsBegin);
    double startt = MPI_Wtime();
    partitionInvoker->partition(partition.data(), graph, target, seed);
    MPI_Barrier(comm);
    double endt = MPI_Wtime();
    struct rusage memoryStatsEnd;
    getrusage(RUSAGE_SELF, &memoryStatsEnd);
    logInfo(rank) << "Actual partitioning done. Now saving stats";

    // counts weighted and unweighted sizes
    std::vector<int> partitionSize(nparts * 4);
    std::vector<int> partitionSizeLocal(nparts * 4);
    for (int i = 0; i < puml->cells().size(); ++i) {
      assert(partition[i] >= 0);
      assert(partition[i] < nparts);
      ++partitionSizeLocal[partition[i]];
    }
    if (vertexweights == nullptr) {
      for (int i = 0; i < nparts; ++i) {
        partitionSizeLocal[i + nparts] = partitionSizeLocal[i];
      }
    } else {
      for (int i = 0; i < puml->cells().size(); ++i) {
        partitionSizeLocal[partition[i] + nparts] += vertexweights[i];
      }
    }

    std::vector<int> detailPartition(nparts * nparts);
    std::vector<int> localDetailPartition(nparts * nparts);

    graph.forEachLocalEdges<int>(partition,
                                 [&](int fid, int lid, const int& g, const int& l, int id) {
                                   if (g != l) {
                                     ++partitionSizeLocal[g + nparts * 2];
                                     if (edgeweights == nullptr) {
                                       ++partitionSizeLocal[g + nparts * 3];
                                       ++localDetailPartition[g * nparts + l];
                                     } else {
                                       partitionSizeLocal[g + nparts * 3] += edgeweights[id];
                                       localDetailPartition[g * nparts + l] += edgeweights[id];
                                     }
                                   }
                                 });

    MPI_Reduce(partitionSizeLocal.data(),
               partitionSize.data(),
               partitionSizeLocal.size(),
               MPI_INT,
               MPI_SUM,
               0,
               comm);
    MPI_Reduce(localDetailPartition.data(),
               detailPartition.data(),
               localDetailPartition.size(),
               MPI_INT,
               MPI_SUM,
               0,
               comm);

    long peakmemstart = memoryStatsBegin.ru_maxrss;
    long peakmemend = memoryStatsEnd.ru_maxrss;

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

    MPI_Gather(
        &memoryStatsBegin.ru_maxrss, 1, MPI_LONG, peakmemrankstart.data(), 1, MPI_LONG, 0, comm);
    MPI_Gather(&memoryStatsEnd.ru_maxrss, 1, MPI_LONG, peakmemrankend.data(), 1, MPI_LONG, 0, comm);

    if (rank == 0) {
      std::vector<nlohmann::json> node_data(nparts);
      size_t ec = 0;
      size_t ecw = 0;
      size_t ps = 0;
      size_t psw = 0;
      size_t pss = 0;
      size_t psws = 0;
      for (int i = 0; i < nparts; ++i) {
        node_data[i]["size"] = partitionSize[i];
        node_data[i]["size_weighted"] = partitionSize[i + nparts];
        node_data[i]["cut"] = partitionSize[i + nparts * 2];
        node_data[i]["cut_weighted"] = partitionSize[i + nparts * 3];

        std::vector<int> detail(detailPartition.begin() + nparts * i,
                                detailPartition.begin() + nparts * (i + 1));
        node_data[i]["cut_weighted_detail"] = detail;

        ec += partitionSize[i + nparts * 2];
        ecw += partitionSize[i + nparts * 3];
        ps = std::max(ps, (size_t)partitionSize[i]);
        psw = std::max(psw, (size_t)partitionSize[i + nparts]);
        pss += partitionSize[i];
        psws += partitionSize[i + nparts];
      }
      ec /= 2;
      ecw /= 2;
      double mi = (double)ps / ((double)pss / (double)nparts);
      double miw = (double)psw / ((double)psws / (double)nparts);
      logDebug(rank) << "End stats:" << ec << ecw << mi << miw << peakmem[0] << peakmem[1]
                     << (endt - startt);

      nlohmann::json output;

      output["cut"] = ec;
      output["cut_weighted"] = ecw;
      output["imbalance"] = mi;
      output["imbalance_weighted"] = miw;
      output["total_vertices"] = graph.globalVertexCount();
      output["total_edges"] = graph.globalEdgeCount();
      output["mpi_size"] = size;
      output["partition_stats"] = node_data;
      output["time"] = endt - startt;
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

      std::string odir = readSetting<std::string>(settings, "output");
      if (odir == "-") {
        // use stdout
        logInfo(rank) << "Results: " << output.dump(4);
      } else {
        size_t hash = std::hash<nlohmann::json>()(output);

        // this is not ideal, but it avoids the dependency on std::filesystem
        std::ofstream outputFile(odir + "/" + std::to_string(hash) + ".json");
        outputFile << output;
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