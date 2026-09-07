// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>
#include <string>

#include <gtest/gtest.h>

#include "Hdf5Reader.h"
#include "PumlTest.h"

namespace {

using namespace puml::test;

auto meshFile() -> std::string { return std::string(PUML_TEST_DATA_DIR) + "/mesh.h5"; }

TEST(Reader, ReadsTheSampleMesh) {
  PUML::TETPUML puml;
#ifdef USE_MPI
  puml.setComm(MPI_COMM_WORLD);
#endif // USE_MPI
  PUML::Hdf5Reader<PUML::TETRAHEDRON> reader(puml);
  reader.open(meshFile() + ":/connect", meshFile() + ":/geometry");
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
  PUML::Hdf5Reader<PUML::TETRAHEDRON> reader(puml);
  reader.open(meshFile() + ":/connect", meshFile() + ":/geometry");
  reader.addData<int>("group", meshFile() + ":/group", PUML::CELL, {});
  reader.addData<int>("boundary", meshFile() + ":/boundary", PUML::CELL, {});
  puml.generateMesh();

  const auto group = puml.data(puml.find<int>("group", PUML::CELL));
  const auto boundary = puml.data(puml.find<int>("boundary", PUML::CELL));

  long inGroupOne = 0;
  long interior = 0;
  long freeSurface = 0;
  long absorbing = 0;
  for (std::size_t i = 0; i < group.size(); ++i) {
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

/// HDF5 converts between the type on disk and the type asked for, so a dataset
/// can be read at a fixed width no matter what width it was written at. That
/// removes the need to pick the value type at run time.
TEST(Reader, ConvertsTheValueTypeOnRead) {
  PUML::TETPUML narrow;
  PUML::TETPUML wide;
#ifdef USE_MPI
  narrow.setComm(MPI_COMM_WORLD);
  wide.setComm(MPI_COMM_WORLD);
#endif // USE_MPI

  PUML::Hdf5Reader<PUML::TETRAHEDRON> narrowReader(narrow);
  PUML::Hdf5Reader<PUML::TETRAHEDRON> wideReader(wide);

  narrowReader.inferSize(PUML::CELL, meshFile() + ":/connect");
  wideReader.inferSize(PUML::CELL, meshFile() + ":/connect");

  // The dataset holds 32 bit integers.
  narrowReader.addData<std::int32_t>("boundary", meshFile() + ":/boundary", PUML::CELL, {});
  wideReader.addData<std::uint64_t>("boundary", meshFile() + ":/boundary", PUML::CELL, {});

  const auto asWritten = narrow.data(narrow.find<std::int32_t>("boundary", PUML::CELL));
  const auto asAsked = wide.data(wide.find<std::uint64_t>("boundary", PUML::CELL));

  EXPECT_EQ(asWritten.size(), asAsked.size());
  for (std::size_t i = 0; i < asWritten.size(); ++i) {
    EXPECT_EQ(static_cast<std::uint64_t>(asWritten[i]), asAsked[i]) << "cell " << i;
  }
}

/// The file says how wide its values are and how many of them sit next to each
/// other, which is what lets a caller tell shapes apart without being told.
TEST(Reader, ReportsHowADatasetIsLaidOut) {
  PUML::TETPUML puml;
#ifdef USE_MPI
  puml.setComm(MPI_COMM_WORLD);
#endif // USE_MPI
  PUML::Hdf5Reader<PUML::TETRAHEDRON> reader(puml);

  const auto connect = reader.datasetShape(meshFile() + ":/connect");
  EXPECT_EQ(connect.valueBytes, 8U);
  ASSERT_EQ(connect.dims.size(), 2U);
  EXPECT_EQ(connect.dims[0], 19183U);
  EXPECT_EQ(connect.dims[1], 4U);

  // The boundary of the sample mesh is written as 32 bit values, one per cell.
  const auto boundary = reader.datasetShape(meshFile() + ":/boundary");
  EXPECT_EQ(boundary.valueBytes, 4U);
  ASSERT_EQ(boundary.dims.size(), 1U);
  EXPECT_EQ(boundary.dims[0], 19183U);

  const auto geometry = reader.datasetShape(meshFile() + ":/geometry");
  EXPECT_EQ(geometry.valueBytes, 8U);
  EXPECT_EQ(geometry.dims[1], 3U);
}

} // namespace
