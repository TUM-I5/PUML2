// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "PumlTest.h"

namespace {

using namespace puml::test;

/// Every cell carries a different number of values, derived from its identity,
/// so that both the count and the values can be checked after they moved.
auto countOf(std::size_t cell) -> PUML::Size { return 1 + (cell % 5); }

void fill(PUML::TETPUML& puml, Split cells) {
  std::vector<PUML::Size> counts(cells.size);
  for (std::size_t i = 0; i < cells.size; ++i) {
    counts[i] = countOf(cells.offset + i);
  }

  const auto handle = puml.allocateRaggedData<unsigned long>("nodes", counts);
  auto values = puml.raggedData(handle);
  for (std::size_t i = 0; i < cells.size; ++i) {
    auto* first = values.begin(i);
    for (PUML::Size j = 0; j < counts[i]; ++j) {
      first[j] = 100 * (cells.offset + i) + j;
    }
  }

  std::vector<unsigned long> identity(cells.size);
  for (std::size_t i = 0; i < cells.size; ++i) {
    identity[i] = cells.offset + i;
  }
  puml.addDataArray<unsigned long>("identity", identity.data(), PUML::CELL, {});
}

/// Checks that every cell still holds the values that belong to it.
void checkValues(const PUML::TETPUML& puml) {
  const auto identity = puml.data(puml.find<unsigned long>("identity", PUML::CELL));
  const auto values = puml.raggedData(puml.findRagged<unsigned long>("nodes"));

  EXPECT_EQ(values.entities(), identity.size());
  for (std::size_t i = 0; i < identity.size(); ++i) {
    const auto cell = identity[i];
    EXPECT_EQ(values.count(i), countOf(cell)) << "cell " << cell;
    const auto* first = values.begin(i);
    for (PUML::Size j = 0; j < values.count(i); ++j) {
      EXPECT_EQ(first[j], 100 * cell + j) << "cell " << cell;
    }
  }
}

TEST(Ragged, HoldsADifferentCountPerCell) {
  const auto mesh = makeCubeMesh(3);
  const auto cells = evenSplit(mesh.numCells, commRank(), commSize());

  PUML::TETPUML puml;
  feed(puml, mesh, cells, evenSplit(mesh.numVertices, commRank(), commSize()));
  fill(puml, cells);

  const auto values = puml.raggedData(puml.findRagged<unsigned long>("nodes"));
  PUML::Size total = 0;
  for (std::size_t i = 0; i < cells.size; ++i) {
    total += countOf(cells.offset + i);
  }
  EXPECT_EQ(values.size(), total);
  checkValues(puml);
}

/// The values follow their cell, and so do their counts.
TEST(Ragged, SurvivesPartitioning) {
  const auto mesh = makeCubeMesh(3);
  const auto cells = evenSplit(mesh.numCells, commRank(), commSize());
  const int procs = commSize();

  PUML::TETPUML puml;
  feed(puml, mesh, cells, evenSplit(mesh.numVertices, commRank(), commSize()));
  fill(puml, cells);

  std::vector<int> target(cells.size);
  for (std::size_t i = 0; i < cells.size; ++i) {
    target[i] = static_cast<int>(i % procs);
  }
  puml.partition(target.data());

  checkValues(puml);

  // Nothing was lost on the way.
  const auto identity = puml.data(puml.find<unsigned long>("identity", PUML::CELL));
  auto all = gather(std::vector<unsigned long>(identity.begin(), identity.end()));
  std::sort(all.begin(), all.end());
  EXPECT_EQ(all.size(), mesh.numCells);
  EXPECT_TRUE(isContiguousFromZero(all));

  puml.generateMesh();
  const auto counts = measure(puml);
  EXPECT_EQ(counts.cells, static_cast<long>(mesh.numCells));
  EXPECT_EQ(counts.euler(), 1);
}

TEST(Ragged, ChecksTheValueTypeAndTheCounts) {
  const auto mesh = makeCubeMesh(2);
  const auto cells = evenSplit(mesh.numCells, commRank(), commSize());

  PUML::TETPUML puml;
  feed(puml, mesh, cells, evenSplit(mesh.numVertices, commRank(), commSize()));
  fill(puml, cells);

  std::string message;
  try {
    static_cast<void>(puml.findRagged<double>("nodes"));
  } catch (const PUML::Error& error) {
    message = error.what();
  }
  EXPECT_NE(message.find("another type"), std::string::npos) << message;

  message.clear();
  try {
    static_cast<void>(puml.allocateRaggedData<double>("bad", std::vector<PUML::Size>{1, 2}));
  } catch (const PUML::Error& error) {
    message = error.what();
  }
  EXPECT_NE(message.find("one count per cell"), std::string::npos) << message;
}

} // namespace
