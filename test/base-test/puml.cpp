// SPDX-FileCopyrightText: 2017 Technical University of Munich
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
 */

#include <mpi.h>

#include "utils/args.h"
#include "utils/logger.h"

#include "PUML.h"
#include "Downward.h"
#include "Neighbor.h"
#include "Partition.h"
#include "PartitionGraph.h"

#include "XdmfWriter/XdmfWriter.h"

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  utils::Args args;
  args.addAdditionalOption("in", "the PUML mesh file");
  args.addAdditionalOption("out", "the output file");

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
  std::string outfile = args.getAdditionalArgument<const char*>("out");

  // Read the mesh
  logInfo(rank) << "Reading mesh";
  puml.open((infile + ":/connect").c_str(), (infile + ":/geometry").c_str());

  logInfo(rank) << "Reading other data";
  puml.addData((infile + ":/group").c_str(), PUML::CELL);

  // Run the partitioning
  logInfo(rank) << "Creating partitions";
  int* partition = new int[puml.numOriginalCells()];

  PUML::TETPartitionGraph graph(puml);
  auto partitioner = PUML::Partition::getPartitioner(PUML::PartitionerType::Parmetis);
  PUML::PartitionTarget target;
  target.setVertexWeightsUniform();
  target.setImbalance(1.01);
  partitioner->partition(partition, graph, target);

  // Redistribute the cells
  logInfo(rank) << "Redistributing cells";
  puml.partition(partition);
  delete[] partition;

  // Generate the mesh information
  logInfo(rank) << "Generating mesh information";
  puml.generateMesh();

  // Write test
  logInfo(rank) << "Writing test file";
  const std::vector<PUML::TETPUML::cell_t>& cells = puml.cells();

  const std::vector<PUML::TETPUML::vertex_t>& vertices = puml.vertices();

  xdmfwriter::XdmfWriter<xdmfwriter::TETRAHEDRON, double> writer(xdmfwriter::POSIX,
                                                                 outfile.c_str());

  std::vector<const char*> cellVars;
  cellVars.push_back("group");
  std::vector<const char*> vertexVars;
  writer.init(cellVars, vertexVars);

  double* coords = new double[vertices.size() * 3];
  for (unsigned int i = 0; i < vertices.size(); i++) {
    memcpy(&coords[i * 3], vertices[i].coordinate(), 3 * sizeof(double));
  }

  unsigned int* cellPoints = new unsigned int[cells.size() * 4];
  for (unsigned int i = 0; i < cells.size(); i++) {
    PUML::Downward::vertices(puml, cells[i], &cellPoints[i * 4]);
  }

  writer.setMesh(cells.size(), cellPoints, vertices.size(), coords);

  writer.addTimeStep(0);

  // Write cell/vertex data
  double* cellData = new double[cells.size()];
  for (unsigned int i = 0; i < cells.size(); i++)
    cellData[i] = puml.cellData(0)[i];

  writer.writeCellData(0, cellData);

  delete[] cellData;

  writer.close();

  logInfo(rank) << "Done";

  MPI_Finalize();

  return 0;
}