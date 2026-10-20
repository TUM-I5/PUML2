// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <functional>
#include <string>

#include <gtest/gtest.h>

#include "Error.h"
#include "PUML.h"
#include "PumlTest.h"

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
