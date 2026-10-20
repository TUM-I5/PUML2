// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <cstddef>
#include <functional>
#include <string>

#include <gtest/gtest.h>

#include "Error.h"
#include "PUML.h"
#include "PumlTest.h"
#include "Types.h"

namespace {

using namespace puml::test;

/// What the call raised on this rank, or nothing if it went through.
auto messageOf(const std::function<void()>& call) -> std::string {
  try {
    call();
  } catch (const PUML::Error& error) {
    return error.what();
  }
  return {};
}

/// Builds the mesh from its cells, split evenly over the ranks, and gives what
/// this rank raised. A defect has to be raised on every rank, also on the ones
/// that do not hold it, or they would wait for the others forever.
auto generate(const CubeMesh& mesh) -> std::string {
  PUML::TETPUML puml;
  feed(puml,
       mesh,
       evenSplit(mesh.numCells, commRank(), commSize()),
       evenSplit(mesh.numVertices, commRank(), commSize()));
  return messageOf([&puml]() { puml.generateMesh(); });
}

/// The rank that a message names as the one that found the defect.
auto foundOn(int rank) -> std::string {
#ifdef USE_MPI
  return "(Found on rank " + std::to_string(rank);
#else  // USE_MPI
  static_cast<void>(rank);
  return {};
#endif // USE_MPI
}

/// The first cell of the last rank, which no other rank holds.
auto cellOfTheLastRank(const CubeMesh& mesh) -> std::size_t {
  return evenSplit(mesh.numCells, commSize() - 1, commSize()).offset;
}

/// A vertex id beyond the vertices of the mesh names a vertex that no rank
/// holds; the rank with the cell has to raise it before it asks for vertices.
TEST(Defect, RaisesAVertexBeyondTheVertices) {
  auto mesh = makeCubeMesh(2);
  const auto cell = cellOfTheLastRank(mesh);
  const auto beyond = mesh.numVertices + 3;
  mesh.connect[(4 * cell) + 1] = beyond;

  const auto message = generate(mesh);
  EXPECT_NE(message.find("cell " + std::to_string(cell) + " names the vertex " +
                         std::to_string(beyond) + ", but the mesh has " +
                         std::to_string(mesh.numVertices) + " vertices"),
            std::string::npos)
      << message;
  EXPECT_NE(message.find(foundOn(commSize() - 1)), std::string::npos) << message;
}

/// Ids that count from 1 name one vertex behind the last one, and the message
/// says what that may mean.
TEST(Defect, RaisesVertexIdsCountedFromOne) {
  auto mesh = makeCubeMesh(2);
  for (auto& vertex : mesh.connect) {
    ++vertex;
  }

  // The first cell to name that vertex is the one the message names.
  const auto namesIt = [&mesh](std::size_t cell) {
    const auto first = mesh.connect.begin() + static_cast<std::ptrdiff_t>(4 * cell);
    return std::find(first, first + 4, mesh.numVertices) != first + 4;
  };
  std::size_t cell = 0;
  while (!namesIt(cell)) {
    ++cell;
  }

  const auto message = generate(mesh);
  EXPECT_NE(message.find("cell " + std::to_string(cell) + " names the vertex " +
                         std::to_string(mesh.numVertices)),
            std::string::npos)
      << message;
  EXPECT_NE(message.find("counted from 1"), std::string::npos) << message;
}

/// An id with all its bits set is what a writer may put for a vertex it does
/// not know.
TEST(Defect, RaisesAVertexIdWithAllBitsSet) {
  auto mesh = makeCubeMesh(2);
  mesh.connect[4 * cellOfTheLastRank(mesh)] = ~PUML::GlobalId{0};

  const auto message = generate(mesh);
  EXPECT_NE(message.find("All bits of the id are set"), std::string::npos) << message;
}

/// A mesh constructed from cells whose vertices were never distributed names
/// vertices that the ranks do not hold.
TEST(Defect, RaisesAVertexThatWasNotDistributed) {
  const auto mesh = makeCubeMesh(2);
  PUML::TETPUML puml;
  feed(puml,
       mesh,
       evenSplit(mesh.numCells, commRank(), commSize()),
       evenSplit(mesh.numVertices, commRank(), commSize()));
  puml.distributeVertices({});

  const auto message = messageOf([&puml]() { puml.constructMesh(puml.connectivity()); });
  EXPECT_NE(message.find("cell 0 names the vertex " + std::to_string(mesh.connect[0]) +
                         ", which this rank does not hold"),
            std::string::npos)
      << message;
  EXPECT_NE(message.find(foundOn(0)), std::string::npos) << message;
}

/// A cell listed twice, right behind itself, so that rank 0 holds both copies:
/// its faces have a third cell there, which the other ranks cannot see.
TEST(Defect, RaisesAFaceWithThreeCellsOnEveryRank) {
  auto mesh = makeCubeMesh(2);
  mesh.connect.insert(mesh.connect.begin() + 4, mesh.connect.begin(), mesh.connect.begin() + 4);
  mesh.numCells += 1;

  const auto message = generate(mesh);
  EXPECT_NE(message.find("more than two cells"), std::string::npos) << message;
  EXPECT_NE(message.find("duplicated cell"), std::string::npos) << message;
  EXPECT_NE(message.find(foundOn(0)), std::string::npos) << message;
}

} // namespace
