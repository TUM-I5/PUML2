// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "PumlTest.h"

namespace {

using namespace puml::test;

auto cubeWithData(PUML::TETPUML& puml, const CubeMesh& mesh, Split cells) -> void {
  feed(puml, mesh, cells, evenSplit(mesh.numVertices, commRank(), commSize()));

  std::vector<std::int32_t> narrow(cells.size);
  std::vector<double> wide(cells.size * 3);
  for (std::size_t i = 0; i < cells.size; ++i) {
    narrow[i] = static_cast<std::int32_t>(cells.offset + i);
    for (std::size_t j = 0; j < 3; ++j) {
      wide[(3 * i) + j] = static_cast<double>((3 * (cells.offset + i)) + j);
    }
  }
  puml.addDataArray<std::int32_t>("narrow", narrow.data(), PUML::CELL, {});
  puml.addDataArray<double>("wide", wide.data(), PUML::CELL, {3});
}

TEST(Handle, KnowsTheShapeOfItsArray) {
  const auto mesh = makeCubeMesh(2);
  const auto cells = evenSplit(mesh.numCells, commRank(), commSize());

  PUML::TETPUML puml;
  cubeWithData(puml, mesh, cells);

  const auto narrow = puml.find<std::int32_t>("narrow", PUML::CELL);
  const auto wide = puml.find<double>("wide", PUML::CELL);

  EXPECT_TRUE(narrow.valid());
  EXPECT_EQ(narrow.type(), PUML::CELL);
  EXPECT_EQ(narrow.elemCount(), 1U);
  EXPECT_EQ(wide.elemCount(), 3U);

  EXPECT_EQ(puml.data(narrow).size(), cells.size);
  EXPECT_EQ(puml.data(wide).size(), 3 * cells.size);
  EXPECT_EQ(puml.data(wide).entities(), cells.size);
}

/// The check that turns a name into a type. A width that happens to match is
/// not enough: int32 and uint32 are different arrays.
TEST(Handle, TellsTheValueTypeApart) {
  const auto mesh = makeCubeMesh(2);
  PUML::TETPUML puml;
  cubeWithData(puml, mesh, evenSplit(mesh.numCells, commRank(), commSize()));

  EXPECT_TRUE(puml.holds<std::int32_t>("narrow", PUML::CELL));
  EXPECT_FALSE(puml.holds<std::uint32_t>("narrow", PUML::CELL));
  EXPECT_FALSE(puml.holds<double>("narrow", PUML::CELL));
  EXPECT_FALSE(puml.holds<std::int32_t>("wide", PUML::CELL));
  EXPECT_TRUE(puml.holds<double>("wide", PUML::CELL));

  // A cell array is not a vertex array.
  EXPECT_FALSE(puml.holds<std::int32_t>("narrow", PUML::VERTEX));

  EXPECT_FALSE(puml.holds<double>("does-not-exist", PUML::CELL));
  EXPECT_FALSE(puml.has("does-not-exist", PUML::CELL));
  EXPECT_TRUE(puml.has("narrow", PUML::CELL));
}

/// Values read through a handle are the values that were written.
TEST(Handle, ReadsWhatWasWritten) {
  const auto mesh = makeCubeMesh(2);
  const auto cells = evenSplit(mesh.numCells, commRank(), commSize());

  PUML::TETPUML puml;
  cubeWithData(puml, mesh, cells);

  const auto narrow = puml.data(puml.find<std::int32_t>("narrow", PUML::CELL));
  const auto wide = puml.data(puml.find<double>("wide", PUML::CELL));

  for (std::size_t i = 0; i < cells.size; ++i) {
    EXPECT_EQ(narrow[i], static_cast<std::int32_t>(cells.offset + i));
    const auto* values = wide.entity(i);
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(values[j], static_cast<double>((3 * (cells.offset + i)) + j));
    }
  }
}

/// A handle stays valid while other arrays come and go.
TEST(Handle, SurvivesFurtherArrays) {
  const auto mesh = makeCubeMesh(2);
  const auto cells = evenSplit(mesh.numCells, commRank(), commSize());

  PUML::TETPUML puml;
  cubeWithData(puml, mesh, cells);

  const auto narrow = puml.find<std::int32_t>("narrow", PUML::CELL);

  std::vector<double> more(cells.size, 1.0);
  puml.addDataArray<double>("more", more.data(), PUML::CELL, {});
  puml.addDataArray<double>("even-more", more.data(), PUML::CELL, {});

  const auto values = puml.data(narrow);
  for (std::size_t i = 0; i < cells.size; ++i) {
    EXPECT_EQ(values[i], static_cast<std::int32_t>(cells.offset + i));
  }
}

/// Vertex arrays are readable in both stages.
TEST(Handle, SeparatesTheVertexStages) {
  const auto mesh = makeCubeMesh(2);
  const auto vertices = evenSplit(mesh.numVertices, commRank(), commSize());

  PUML::TETPUML puml;
  feed(puml, mesh, evenSplit(mesh.numCells, commRank(), commSize()), vertices);

  const auto geometry = puml.find<double>("geometry", PUML::VERTEX);
  EXPECT_EQ(puml.data(geometry).entities(), vertices.size);

  puml.generateMesh();
  EXPECT_EQ(puml.distributedData(geometry).entities(), puml.vertices().size());
  EXPECT_EQ(puml.data(geometry).entities(), vertices.size);
}

} // namespace
