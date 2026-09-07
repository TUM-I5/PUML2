// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <array>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "Downward.h"
#include "PumlTest.h"
#include "Upward.h"

namespace {

using namespace puml::test;

/// Builds a cube mesh in memory and checks the topology PUML derives from it.
void checkCubeMesh(int n, Split (*split)(std::size_t, int, int)) {
  const int rank = commRank();
  const int procs = commSize();
  const auto mesh = makeCubeMesh(n);

  PUML::TETPUML puml;
  feed(puml, mesh, split(mesh.numCells, rank, procs), split(mesh.numVertices, rank, procs));
  puml.generateMesh();

  const auto counts = measure(puml);

  EXPECT_EQ(counts.cells, static_cast<long>(mesh.numCells));
  EXPECT_EQ(counts.vertices, static_cast<long>(mesh.numVertices));
  EXPECT_EQ(counts.boundaryFaces, mesh.numBoundaryFaces());
  EXPECT_EQ(counts.unusedFaces, 0);

  // A tetrahedral mesh filling a ball is contractible.
  EXPECT_EQ(counts.euler(), 1);

  // Every face is used by four cell-face slots in total, counting each interior
  // face twice.
  EXPECT_EQ(2 * counts.faces - counts.boundaryFaces, 4 * counts.cells);

  // The global ids of each entity kind are unique and gapless.
  for (const auto& gids :
       {allGids(puml.faces()), allGids(puml.edges()), allGids(puml.vertices())}) {
    EXPECT_TRUE(isContiguousFromZero(gids));
  }
}

TEST(Mesh, InMemoryCubeEvenSplit) {
  checkCubeMesh(2, evenSplit);
  checkCubeMesh(4, evenSplit);
}

TEST(Mesh, InMemoryCubeLopsidedSplit) {
  checkCubeMesh(2, lopsidedSplit);
  checkCubeMesh(4, lopsidedSplit);
}

TEST(Mesh, UpwardListsAreSortedAndFreeOfDuplicates) {
  const auto mesh = makeCubeMesh(3);
  PUML::TETPUML puml;
  feed(puml,
       mesh,
       evenSplit(mesh.numCells, commRank(), commSize()),
       evenSplit(mesh.numVertices, commRank(), commSize()));
  puml.generateMesh();

  for (const auto& vertex : puml.vertices()) {
    std::vector<PUML::LocalId> edges;
    PUML::Upward::edges(puml, vertex, edges);
    EXPECT_TRUE(std::is_sorted(edges.begin(), edges.end()));
    EXPECT_EQ(std::set<PUML::LocalId>(edges.begin(), edges.end()).size(), edges.size());
    EXPECT_FALSE(edges.empty());
  }

  for (const auto& edge : puml.edges()) {
    std::vector<PUML::LocalId> faces;
    PUML::Upward::faces(puml, edge, faces);
    EXPECT_TRUE(std::is_sorted(faces.begin(), faces.end()));
    EXPECT_EQ(std::set<PUML::LocalId>(faces.begin(), faces.end()).size(), faces.size());
  }
}

/// Every cell reached upwards from an edge has to be a real cell that the edge
/// belongs to.
/// A two-dimensional mesh: its faces are edges, it has no edge entities, and
/// its vertices still carry three coordinates.
TEST(Mesh, InMemorySquare) {
  for (const int n : {2, 4}) {
    const auto mesh = makeSquareMesh(n);
    PUML::PUML<PUML::TRIANGLE> puml;
    feed(puml,
         mesh,
         evenSplit(mesh.numCells, commRank(), commSize()),
         evenSplit(mesh.numVertices, commRank(), commSize()));
    puml.generateMesh();

    const auto counts = measure(puml);

    EXPECT_EQ(counts.cells, static_cast<long>(mesh.numCells));
    EXPECT_EQ(counts.vertices, static_cast<long>(mesh.numVertices));
    EXPECT_EQ(counts.faces, mesh.numFaces());
    EXPECT_EQ(counts.boundaryFaces, mesh.numBoundaryFaces());
    EXPECT_EQ(counts.unusedFaces, 0);
    EXPECT_EQ(counts.edges, 0) << "a two-dimensional mesh has no edge entities";

    // Euler in two dimensions, where a face is an edge of the mesh.
    EXPECT_EQ(counts.vertices - counts.faces + counts.cells, 1);

    // A triangle has three edges, an interior one shared by two cells.
    EXPECT_EQ(2 * counts.faces - counts.boundaryFaces, 3 * counts.cells);

    EXPECT_TRUE(isContiguousFromZero(allGids(puml.faces())));
    EXPECT_TRUE(isContiguousFromZero(allGids(puml.vertices())));

    // The third coordinate survives the distribution.
    for (const auto& vertex : puml.vertices()) {
      EXPECT_DOUBLE_EQ(vertex.coordinate()[2], 0.0);
      EXPECT_GE(vertex.coordinate()[0], 0.0);
      EXPECT_LE(vertex.coordinate()[0], static_cast<double>(n));
    }
  }
}

/// Walking up from a vertex in two dimensions goes straight to the cells.
TEST(Mesh, TwoDimensionalUpward) {
  const auto mesh = makeSquareMesh(3);
  PUML::PUML<PUML::TRIANGLE> puml;
  feed(puml,
       mesh,
       evenSplit(mesh.numCells, commRank(), commSize()),
       evenSplit(mesh.numVertices, commRank(), commSize()));
  puml.generateMesh();

  for (const auto& vertex : puml.vertices()) {
    std::vector<PUML::LocalId> cells;
    PUML::Upward::cells(puml, vertex, cells);

    EXPECT_FALSE(cells.empty());
    EXPECT_TRUE(std::is_sorted(cells.begin(), cells.end()));
    for (const auto cell : cells) {
      EXPECT_NE(cell, PUML::InvalidLocalId);
      EXPECT_LT(cell, puml.cells().size());
    }
  }
}

/// The same invariants on a hexahedral mesh.
TEST(Mesh, InMemoryHexCube) {
  for (const int n : {2, 3}) {
    const auto mesh = makeHexCubeMesh(n);
    PUML::HEXPUML puml;
    feed(puml,
         mesh,
         evenSplit(mesh.numCells, commRank(), commSize()),
         evenSplit(mesh.numVertices, commRank(), commSize()));
    puml.generateMesh();

    const auto counts = measure(puml);

    EXPECT_EQ(counts.cells, static_cast<long>(mesh.numCells));
    EXPECT_EQ(counts.vertices, static_cast<long>(mesh.numVertices));
    EXPECT_EQ(counts.boundaryFaces, mesh.numBoundaryFaces());
    EXPECT_EQ(counts.unusedFaces, 0);
    EXPECT_EQ(counts.euler(), 1);

    // A hexahedron has six faces, an interior one shared by two cells.
    EXPECT_EQ(2 * counts.faces - counts.boundaryFaces, 6 * counts.cells);
    EXPECT_EQ(counts.faces, 3L * n * n * (n + 1));
    EXPECT_EQ(counts.edges, 3L * n * (n + 1) * (n + 1));

    for (const auto& gids : {allGids(puml.faces()), allGids(puml.edges())}) {
      EXPECT_TRUE(isContiguousFromZero(gids));
    }
  }
}

TEST(Mesh, UpwardCellsAreValid) {
  const auto mesh = makeCubeMesh(3);
  PUML::TETPUML puml;
  feed(puml,
       mesh,
       evenSplit(mesh.numCells, commRank(), commSize()),
       evenSplit(mesh.numVertices, commRank(), commSize()));
  puml.generateMesh();

  for (const auto& edge : puml.edges()) {
    std::vector<PUML::LocalId> cells;
    PUML::Upward::cells(puml, edge, cells);

    EXPECT_FALSE(cells.empty());
    for (const auto cell : cells) {
      EXPECT_NE(cell, PUML::InvalidLocalId);
      EXPECT_LT(cell, puml.cells().size());
    }
  }

  for (const auto& face : puml.faces()) {
    std::array<PUML::LocalId, 2> adjacent{};
    PUML::Upward::cells(puml, face, adjacent.data());
    EXPECT_NE(adjacent[0], PUML::InvalidLocalId);
    EXPECT_LT(adjacent[0], puml.cells().size());
    if (adjacent[1] != PUML::InvalidLocalId) {
      EXPECT_LT(adjacent[1], puml.cells().size());
      EXPECT_NE(adjacent[0], adjacent[1]);
    }
  }
}

/// Walking down to a face and back up has to find the cell again.
TEST(Mesh, DownwardAndUpwardAgree) {
  const auto mesh = makeCubeMesh(3);
  PUML::TETPUML puml;
  feed(puml,
       mesh,
       evenSplit(mesh.numCells, commRank(), commSize()),
       evenSplit(mesh.numVertices, commRank(), commSize()));
  puml.generateMesh();

  for (unsigned int cell = 0; cell < puml.cells().size(); ++cell) {
    std::array<PUML::LocalId, 4> faces{};
    PUML::Downward::faces(puml, puml.cells()[cell], faces.data());

    for (const auto face : faces) {
      std::array<PUML::LocalId, 2> adjacent{};
      PUML::Upward::cells(puml, puml.faces()[face], adjacent.data());
      EXPECT_TRUE(adjacent[0] == cell || adjacent[1] == cell)
          << "cell " << cell << " is missing from face " << face;
    }
  }
}

/// Rebuilding the mesh from the same input has to give the same result.
TEST(Mesh, GenerateMeshIsRepeatable) {
  const auto mesh = makeCubeMesh(3);
  PUML::TETPUML puml;
  feed(puml,
       mesh,
       evenSplit(mesh.numCells, commRank(), commSize()),
       evenSplit(mesh.numVertices, commRank(), commSize()));

  puml.generateMesh();
  const auto first = measure(puml);
  const auto firstFaceGids = allGids(puml.faces());

  puml.generateMesh();
  const auto second = measure(puml);

  EXPECT_EQ(first.cells, second.cells);
  EXPECT_EQ(first.faces, second.faces);
  EXPECT_EQ(first.edges, second.edges);
  EXPECT_EQ(first.vertices, second.vertices);
  EXPECT_EQ(firstFaceGids, allGids(puml.faces()));
}

} // namespace
