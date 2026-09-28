// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "PumlTest.h"

#include "FaceIterator.h"
#include "PUML.h"
#include "Topology.h"
#include "Types.h"
#include "Upward.h"

namespace {

using namespace puml::test;

using Neighbors = std::vector<std::pair<unsigned long, unsigned long>>;

/// What a walk over the faces of a cube mesh has to find: every cell together
/// with each of its neighbours, and the cell of every boundary face.
struct Expected {
  Neighbors neighbors;
  std::vector<unsigned long> boundary;
};

auto expectedOf(const CubeMesh& mesh) -> Expected {
  std::map<std::array<unsigned long, 3>, std::vector<unsigned long>> cellsOfFace;
  for (std::size_t cell = 0; cell < mesh.numCells; ++cell) {
    const auto* vertices = mesh.connect.data() + (4 * cell);
    // Each face of a tetrahedron leaves out one of its vertices.
    for (std::size_t left = 0; left < 4; ++left) {
      std::array<unsigned long, 3> face{};
      std::size_t next = 0;
      for (std::size_t vertex = 0; vertex < 4; ++vertex) {
        if (vertex != left) {
          face[next++] = vertices[vertex];
        }
      }
      std::sort(face.begin(), face.end());
      cellsOfFace[face].push_back(cell);
    }
  }

  Expected expected;
  for (const auto& [face, cells] : cellsOfFace) {
    if (cells.size() == 2) {
      expected.neighbors.emplace_back(cells[0], cells[1]);
      expected.neighbors.emplace_back(cells[1], cells[0]);
    } else {
      expected.boundary.push_back(cells[0]);
    }
  }
  std::sort(expected.neighbors.begin(), expected.neighbors.end());
  std::sort(expected.boundary.begin(), expected.boundary.end());
  return expected;
}

/// Records what the handlers of one walk over the faces are handed, and checks
/// on the way that each face and cell they get belong together.
class Walk {
  public:
  explicit Walk(const PUML::TETPUML& puml) : m_puml(puml) {}

  /// A face handler that gets the value of the cell on the far side.
  auto farSide() {
    return
        [this](int fid, int cid, const unsigned long& farValue) { neighbor(fid, cid, farValue); };
  }

  /// A face handler that gets the value of the cell on the near side as well.
  template <typename S>
  auto bothSides() {
    return [this](int fid, int cid, const unsigned long& farValue, const S& nearValue) {
      EXPECT_EQ(nearValue, static_cast<S>(gid(cid)));
      neighbor(fid, cid, farValue);
    };
  }

  auto boundary() {
    return [this](int fid, int cid) {
      const auto& face = m_puml.faces()[fid];
      const auto cells = PUML::Upward::cells(m_puml, face);
      EXPECT_EQ(cells[0], static_cast<PUML::LocalId>(cid));
      EXPECT_EQ(cells[1], PUML::InvalidLocalId);
      EXPECT_FALSE(face.isShared());
      m_boundary.push_back(gid(cid));
    };
  }

  /// Checks what the walk found on all ranks together against the mesh.
  void check(const Expected& expected, bool withBoundary, const std::string& which) const {
    const auto nearGids = gather(m_near);
    const auto farValues = gather(m_far);
    Neighbors neighbors;
    for (std::size_t i = 0; i < nearGids.size(); ++i) {
      neighbors.emplace_back(nearGids[i], farValues[i]);
    }
    std::sort(neighbors.begin(), neighbors.end());
    EXPECT_EQ(neighbors, expected.neighbors) << which;

    auto boundary = gather(m_boundary);
    std::sort(boundary.begin(), boundary.end());
    if (withBoundary) {
      EXPECT_EQ(boundary, expected.boundary) << which;
    } else {
      EXPECT_TRUE(boundary.empty()) << which;
    }
  }

  private:
  void neighbor(int fid, int cid, unsigned long farValue) {
    const auto cells = PUML::Upward::cells(m_puml, m_puml.faces()[fid]);
    EXPECT_TRUE(cells[0] == static_cast<PUML::LocalId>(cid) ||
                cells[1] == static_cast<PUML::LocalId>(cid));
    m_near.push_back(gid(cid));
    m_far.push_back(farValue);
  }

  [[nodiscard]] auto gid(int cid) const -> unsigned long { return m_puml.cells()[cid].gid(); }

  const PUML::TETPUML& m_puml;
  std::vector<unsigned long> m_near;
  std::vector<unsigned long> m_far;
  std::vector<unsigned long> m_boundary;
};

/// Every overload of forEach hands each interior face to the face handler once
/// from either side, with the value of the cell on the other side, which may be
/// held by another rank, and each boundary face to the boundary handler.
TEST(FaceIterator, EveryOverloadVisitsEveryFace) {
  const auto mesh = makeCubeMesh(2);
  const auto expected = expectedOf(mesh);

  PUML::TETPUML puml;
  feed(puml,
       mesh,
       evenSplit(mesh.numCells, commRank(), commSize()),
       evenSplit(mesh.numVertices, commRank(), commSize()));
  puml.generateMesh();

  // The value of a cell is its global id, as unsigned long for the far side
  // and as long where the near side has a type of its own.
  std::vector<unsigned long> gids(puml.cells().size());
  std::vector<long> nearGids(puml.cells().size());
  for (std::size_t i = 0; i < gids.size(); ++i) {
    gids[i] = puml.cells()[i].gid();
    nearGids[i] = static_cast<long>(gids[i]);
  }
  const auto gidOf = [&gids](int /*fid*/, int cid) { return gids[cid]; };
  const auto nearGidOf = [&nearGids](int /*fid*/, int cid) { return nearGids[cid]; };

  // With and without the sparse exchange between the ranks.
  for (const bool sparse : {true, false}) {
    PUML::FaceIterator<PUML::TETRAHEDRON> iterator(puml, sparse);
    const std::string mode = sparse ? " (sparse)" : " (dense)";

    {
      Walk walk(puml);
      iterator.forEach(gids, walk.bothSides<unsigned long>());
      walk.check(expected, false, "vector" + mode);
    }
    {
      Walk walk(puml);
      iterator.forEach(gids, walk.bothSides<unsigned long>(), walk.boundary());
      walk.check(expected, true, "vector, boundary" + mode);
    }
    {
      Walk walk(puml);
      iterator.forEach(gids, nearGids, walk.bothSides<long>());
      walk.check(expected, false, "two vectors" + mode);
    }
    {
      Walk walk(puml);
      iterator.forEach(gids, nearGids, walk.bothSides<long>(), walk.boundary());
      walk.check(expected, true, "two vectors, boundary" + mode);
    }
    {
      Walk walk(puml);
      iterator.forEach(gids, walk.farSide());
      walk.check(expected, false, "vector, far side" + mode);
    }
    {
      Walk walk(puml);
      iterator.forEach(gids, walk.farSide(), walk.boundary());
      walk.check(expected, true, "vector, far side, boundary" + mode);
    }
    {
      Walk walk(puml);
      iterator.forEach(gids.data(), walk.bothSides<unsigned long>());
      walk.check(expected, false, "pointer" + mode);
    }
    {
      Walk walk(puml);
      iterator.forEach(gids.data(), walk.bothSides<unsigned long>(), walk.boundary());
      walk.check(expected, true, "pointer, boundary" + mode);
    }
    {
      Walk walk(puml);
      iterator.forEach(gids.data(), nearGids.data(), walk.bothSides<long>());
      walk.check(expected, false, "two pointers" + mode);
    }
    {
      Walk walk(puml);
      iterator.forEach(gids.data(), nearGids.data(), walk.bothSides<long>(), walk.boundary());
      walk.check(expected, true, "two pointers, boundary" + mode);
    }
    {
      Walk walk(puml);
      iterator.forEach(gids.data(), walk.farSide());
      walk.check(expected, false, "pointer, far side" + mode);
    }
    {
      Walk walk(puml);
      iterator.forEach(gids.data(), walk.farSide(), walk.boundary());
      walk.check(expected, true, "pointer, far side, boundary" + mode);
    }
    {
      Walk walk(puml);
      iterator.forEach<unsigned long>(gidOf, walk.bothSides<unsigned long>());
      walk.check(expected, false, "handler" + mode);
    }
    {
      Walk walk(puml);
      iterator.forEach<unsigned long>(gidOf, walk.bothSides<unsigned long>(), walk.boundary());
      walk.check(expected, true, "handler, boundary" + mode);
    }
    {
      Walk walk(puml);
      iterator.forEach<unsigned long, long>(gidOf, nearGidOf, walk.bothSides<long>());
      walk.check(expected, false, "two handlers" + mode);
    }
    {
      Walk walk(puml);
      iterator.forEach<unsigned long, long>(
          gidOf, nearGidOf, walk.bothSides<long>(), walk.boundary());
      walk.check(expected, true, "two handlers, boundary" + mode);
    }
    {
      Walk walk(puml);
      iterator.forEach<unsigned long>(gidOf, walk.farSide());
      walk.check(expected, false, "handler, far side" + mode);
    }
    {
      // The handlers by name as well as the temporaries above.
      Walk walk(puml);
      auto farSide = walk.farSide();
      auto boundary = walk.boundary();
      iterator.forEach<unsigned long>(gidOf, farSide, boundary);
      walk.check(expected, true, "handler, far side, boundary" + mode);
    }
  }
}

} // namespace
