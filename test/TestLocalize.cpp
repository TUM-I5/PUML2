// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "Downward.h"
#include "PumlTest.h"

namespace {

using namespace puml::test;

/// The translated array has to agree with the connectivity the mesh was built
/// from.
TEST(Localize, AgreesWithTheConstructedCells) {
  const auto mesh = makeCubeMesh(3);
  const auto cells = evenSplit(mesh.numCells, commRank(), commSize());

  PUML::TETPUML puml;
  feed(puml, mesh, cells, evenSplit(mesh.numVertices, commRank(), commSize()));
  puml.generateMesh();

  const auto localHandle = puml.localize(puml.connectivity(), "local-connectivity");
  const auto local = puml.data(localHandle);

  for (std::size_t i = 0; i < puml.cells().size(); ++i) {
    std::array<PUML::LocalId, 4> expected{};
    PUML::Downward::vertices(puml, puml.cells()[i], expected.data());
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_EQ(local[(4 * i) + j], expected[j]) << "cell " << i;
    }
  }
}

/// Two index arrays over the same vertices: one drives the topology, the other
/// is translated alongside it. This is what makes a second mesh object
/// unnecessary.
TEST(Localize, HandlesASecondIndexArray) {
  const auto mesh = makeCubeMesh(3);
  const auto cells = evenSplit(mesh.numCells, commRank(), commSize());

  PUML::TETPUML puml;
  feed(puml, mesh, cells, evenSplit(mesh.numVertices, commRank(), commSize()));
  puml.addDataArray<unsigned long>(
      "topology", mesh.connect.data() + 4 * cells.offset, PUML::CELL, {4});

  puml.distributeVertices({puml.connectivity(), puml.find<PUML::GlobalId>("topology", PUML::CELL)});
  puml.constructGeometry(puml.geometry());
  puml.constructMesh(puml.find<PUML::GlobalId>("topology", PUML::CELL));

  const auto localHandle = puml.localize(puml.connectivity(), "local-connectivity");
  const auto local = puml.data(localHandle);

  // Both arrays hold the same vertices here, so the translation has to
  // reproduce the cells the topology was built from.
  for (std::size_t i = 0; i < puml.cells().size(); ++i) {
    std::array<PUML::LocalId, 4> expected{};
    PUML::Downward::vertices(puml, puml.cells()[i], expected.data());
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_EQ(local[(4 * i) + j], expected[j]) << "cell " << i;
    }
  }

  const auto counts = measure(puml);
  EXPECT_EQ(counts.cells, static_cast<long>(mesh.numCells));
  EXPECT_EQ(counts.euler(), 1);
}

} // namespace
