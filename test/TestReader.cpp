// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <string>

#include <gtest/gtest.h>

#include "PumlTest.h"

namespace {

using namespace puml::test;

auto meshFile() -> std::string { return std::string(PUML_TEST_DATA_DIR) + "/mesh.h5"; }

TEST(Reader, ReadsTheSampleMesh) {
  PUML::TETPUML puml;
#ifdef USE_MPI
  puml.setComm(MPI_COMM_WORLD);
#endif // USE_MPI
  puml.open(meshFile() + ":/connect", meshFile() + ":/geometry");
  puml.generateMesh();

  const auto counts = measure(puml);

  EXPECT_EQ(counts.cells, 19183);
  EXPECT_EQ(counts.vertices, 3454);
  EXPECT_EQ(counts.faces, 39022);
  EXPECT_EQ(counts.edges, 23292);
  EXPECT_EQ(counts.euler(), 1);
  EXPECT_EQ(counts.unusedFaces, 0);
  EXPECT_EQ(2 * counts.faces - counts.boundaryFaces, 4 * counts.cells);

  EXPECT_TRUE(isContiguousFromZero(allGids(puml.faces())));
  EXPECT_TRUE(isContiguousFromZero(allGids(puml.edges())));
  EXPECT_TRUE(isContiguousFromZero(allGids(puml.vertices())));
}

/// Cell data read from the file keeps its cell.
TEST(Reader, CellDataFollowsTheCells) {
  PUML::TETPUML puml;
#ifdef USE_MPI
  puml.setComm(MPI_COMM_WORLD);
#endif // USE_MPI
  puml.open(meshFile() + ":/connect", meshFile() + ":/geometry");
  puml.addData<int>("group", meshFile() + ":/group", PUML::CELL, {});
  puml.addData<int>("boundary", meshFile() + ":/boundary", PUML::CELL, {});
  puml.generateMesh();

  const auto* group = reinterpret_cast<const int*>(puml.cellData("group"));
  const auto* boundary = reinterpret_cast<const int*>(puml.cellData("boundary"));

  long inGroupOne = 0;
  long interior = 0;
  long freeSurface = 0;
  long absorbing = 0;
  for (std::size_t i = 0; i < puml.numOriginalCells(); ++i) {
    inGroupOne += static_cast<long>(group[i] == 1);
    interior += static_cast<long>(boundary[i] == 0);
    freeSurface += static_cast<long>(boundary[i] == 1);
    absorbing += static_cast<long>(boundary[i] == 3);
  }

  // The sample mesh is one group, and its boundary conditions are distributed
  // over the cells independently of how many ranks read it.
  EXPECT_EQ(globalSum(inGroupOne), 19183);
  EXPECT_EQ(globalSum(interior), 15807);
  EXPECT_EQ(globalSum(freeSurface), 822);
  EXPECT_EQ(globalSum(absorbing), 1990);
}

} // namespace
