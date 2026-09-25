// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * Compares the partitioners the build was configured with, on a generated cube
 * mesh or on a mesh from a file.
 *
 * Usage: puml-partition-benchmark [-n <cells per side>] [-m <file>] [-i <imbalance>]
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include "Hdf5Reader.h"
#include "Partition.h"
#include "PartitionGraph.h"
#include "PartitionTarget.h"
#include "PumlTest.h"

namespace {

using namespace puml::test;
using Clock = std::chrono::steady_clock;

struct Options {
  int n{20};
  std::string mesh;
  double imbalance{0.05};
};

auto parse(int argc, char** argv) -> Options {
  Options options;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    if (flag == "-n") {
      options.n = std::atoi(argv[i + 1]);
    } else if (flag == "-m") {
      options.mesh = argv[i + 1];
    } else if (flag == "-i") {
      options.imbalance = std::atof(argv[i + 1]);
    }
  }
  return options;
}

auto available() -> std::vector<std::pair<std::string, PUML::PartitionerType>> {
  std::vector<std::pair<std::string, PUML::PartitionerType>> partitioners{
      {"None", PUML::PartitionerType::None}};
#ifdef USE_PARMETIS
  partitioners.emplace_back("Parmetis", PUML::PartitionerType::Parmetis);
  partitioners.emplace_back("ParmetisGeometric", PUML::PartitionerType::ParmetisGeometric);
#endif // USE_PARMETIS
#ifdef USE_PTSCOTCH
  partitioners.emplace_back("PtScotch", PUML::PartitionerType::PtScotch);
  partitioners.emplace_back("PtScotchQuality", PUML::PartitionerType::PtScotchQuality);
  partitioners.emplace_back("PtScotchSpeed", PUML::PartitionerType::PtScotchSpeed);
#endif // USE_PTSCOTCH
#ifdef USE_PARHIP
  partitioners.emplace_back("ParHIPFastMesh", PUML::PartitionerType::ParHIPFastMesh);
  partitioners.emplace_back("ParHIPUltrafastMesh", PUML::PartitionerType::ParHIPUltrafastMesh);
#endif // USE_PARHIP
  return partitioners;
}

/// Fills the mesh from a file, or from a generated cube if no file was given.
void load(PUML::TETPUML& puml, const Options& options, const CubeMesh& cube) {
  if (options.mesh.empty()) {
    feed(puml,
         cube,
         evenSplit(cube.numCells, commRank(), commSize()),
         evenSplit(cube.numVertices, commRank(), commSize()));
  } else {
    PUML::Hdf5Reader<PUML::TETRAHEDRON> reader(puml);
    reader.open(options.mesh + ":/connect", options.mesh + ":/geometry");
  }
}

} // namespace

auto main(int argc, char** argv) -> int {
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
#endif // USE_MPI

  const auto options = parse(argc, argv);
  const auto cube = options.mesh.empty() ? makeCubeMesh(options.n) : CubeMesh{};

  if (commRank() == 0) {
    std::printf("%d ranks, imbalance %.3f\n", commSize(), options.imbalance);
    std::printf("%-22s %10s %10s %12s %12s\n",
                "partitioner",
                "time [ms]",
                "imbalance",
                "shared faces",
                "cells");
  }

  for (const auto& [name, type] : available()) {
    PUML::TETPUML puml;
    load(puml, options, cube);
    puml.generateMesh();

    PUML::TETPartitionGraph graph(puml);
    PUML::PartitionTarget target;
    target.setPartitionCount(commSize());
    target.setImbalance(options.imbalance);

    auto partitioner = PUML::TETPartition::getPartitioner(type);

    const auto before = Clock::now();
    const auto part = partitioner->partition(graph, target);
    const auto after = Clock::now();

    puml.partition(part.data());
    puml.generateMesh();

    // What the partitioning is meant to keep small: the faces that end up on a
    // rank boundary, and how unevenly the cells are spread.
    long shared = 0;
    for (const auto& face : puml.faces()) {
      shared += static_cast<long>(!face.shared().empty());
    }
    const auto local = static_cast<long>(puml.numOriginalCells());
    const long total = globalSum(local);
    long largest = local;
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &largest, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
#endif // USE_MPI

    const double milliseconds = std::chrono::duration<double, std::milli>(after - before).count();
    if (commRank() == 0) {
      const double average = static_cast<double>(total) / commSize();
      std::printf("%-22s %10.1f %10.3f %12ld %12ld\n",
                  name.c_str(),
                  milliseconds,
                  (largest / average) - 1.0,
                  globalSum(shared) / 2,
                  total);
    } else {
      static_cast<void>(globalSum(shared));
    }
  }

#ifdef USE_MPI
  MPI_Finalize();
#endif // USE_MPI
  return 0;
}
