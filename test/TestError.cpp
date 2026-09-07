// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "Error.h"
#include "PumlTest.h"

namespace {

using namespace puml::test;

/// Says what it found, so the message points at the mesh rather than at PUML.
auto messageOf(const std::function<void()>& call) -> std::string {
  try {
    call();
  } catch (const PUML::Error& error) {
    return error.what();
  }
  return {};
}

/// A cell listed twice gives its faces a third neighbour.
TEST(Error, ReportsAFaceWithThreeCells) {
  if (commSize() != 1) {
    GTEST_SKIP() << "the defect is local to one rank";
  }

  auto mesh = makeCubeMesh(2);
  // Repeat the first cell, so its four faces are claimed by two cells that are
  // the same and by their real neighbours.
  mesh.connect.insert(mesh.connect.end(), mesh.connect.begin(), mesh.connect.begin() + 4);
  mesh.numCells += 1;

  PUML::TETPUML puml;
  feed(puml, mesh, {0, mesh.numCells}, {0, mesh.numVertices});

  const auto message = messageOf([&puml]() { puml.generateMesh(); });
  EXPECT_NE(message.find("more than two cells"), std::string::npos) << message;
  EXPECT_NE(message.find("duplicated cell"), std::string::npos) << message;
}

TEST(Error, ReportsAnUnknownDataArray) {
  const auto mesh = makeCubeMesh(2);
  PUML::TETPUML puml;
  feed(puml,
       mesh,
       evenSplit(mesh.numCells, commRank(), commSize()),
       evenSplit(mesh.numVertices, commRank(), commSize()));

  const auto message =
      messageOf([&puml]() { static_cast<void>(puml.find<double>("not-there", PUML::CELL)); });
  EXPECT_NE(message.find("not-there"), std::string::npos) << message;
}

TEST(Error, ReportsAWrongValueType) {
  const auto mesh = makeCubeMesh(2);
  PUML::TETPUML puml;
  feed(puml,
       mesh,
       evenSplit(mesh.numCells, commRank(), commSize()),
       evenSplit(mesh.numVertices, commRank(), commSize()));

  const auto message = messageOf(
      [&puml]() { static_cast<void>(puml.find<std::int32_t>("geometry", PUML::VERTEX)); });
  EXPECT_NE(message.find("geometry"), std::string::npos) << message;
  EXPECT_NE(message.find("another type"), std::string::npos) << message;
}

TEST(Error, ReportsDataAddedBeforeTheEntityCount) {
  PUML::TETPUML puml;
#ifdef USE_MPI
  puml.setComm(MPI_COMM_WORLD);
#endif // USE_MPI

  const std::vector<double> values(4, 0.0);
  const auto message = messageOf(
      [&puml, &values]() { puml.addDataArray<double>("x", values.data(), PUML::CELL, {}); });
  EXPECT_NE(message.find("setSize"), std::string::npos) << message;
}

/// Handing a construction step the wrong kind of array is caught where it is
/// looked up, not deep inside the mesh construction.
TEST(Error, ReportsAWrongArrayForConstruction) {
  const auto mesh = makeCubeMesh(2);
  const auto cells = evenSplit(mesh.numCells, commRank(), commSize());

  PUML::TETPUML puml;
  feed(puml, mesh, cells, evenSplit(mesh.numVertices, commRank(), commSize()));

  const std::vector<int> groups(cells.size, 1);
  puml.addDataArray<int>("group", groups.data(), PUML::CELL, {});

  const auto message =
      messageOf([&puml]() { puml.constructMesh(puml.find<PUML::GlobalId>("group", PUML::CELL)); });
  EXPECT_NE(message.find("group"), std::string::npos) << message;
  EXPECT_NE(message.find("another type"), std::string::npos) << message;
}

/// The steps that rewrite an array check what it holds, the same way the ones
/// that build the mesh do.
TEST(Error, ReportsAWrongArrayForLocalize) {
  const auto mesh = makeCubeMesh(2);
  const auto cells = evenSplit(mesh.numCells, commRank(), commSize());

  PUML::TETPUML puml;
  feed(puml, mesh, cells, evenSplit(mesh.numVertices, commRank(), commSize()));

  const std::vector<int> groups(cells.size, 1);
  puml.addDataArray<int>("group", groups.data(), PUML::CELL, {});
  puml.generateMesh();

  const auto message = messageOf([&puml]() {
    static_cast<void>(puml.localize(puml.find<PUML::GlobalId>("group", PUML::CELL), "localized"));
  });
  EXPECT_NE(message.find("group"), std::string::npos) << message;
  EXPECT_NE(message.find("another type"), std::string::npos) << message;
}

TEST(Error, ReportsAWrongArrayForIdentify) {
  const auto mesh = makeCubeMesh(2);
  const auto vertices = evenSplit(mesh.numVertices, commRank(), commSize());

  PUML::TETPUML puml;
  feed(puml, mesh, evenSplit(mesh.numCells, commRank(), commSize()), vertices);

  const std::vector<double> weights(vertices.size, 1.0);
  puml.addDataArray<double>("weights", weights.data(), PUML::VERTEX, {});
  puml.generateMesh();

  const auto message = messageOf([&puml]() {
    puml.identify(puml.connectivity(), puml.find<PUML::GlobalId>("weights", PUML::VERTEX));
  });
  EXPECT_NE(message.find("weights"), std::string::npos) << message;
  EXPECT_NE(message.find("another type"), std::string::npos) << message;
}

} // namespace
