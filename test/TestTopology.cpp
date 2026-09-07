// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include "Numbering.h"
#include "Topology.h"

namespace {

using PUML::internal::Numbering;
using PUML::internal::Topology;

template <PUML::TopoType Topo>
constexpr auto faceVertexCountMatches() -> bool {
  return sizeof(typename Numbering<Topo>::face_t) / sizeof(unsigned int) ==
         Topology<Topo>::facevertices();
}

static_assert(faceVertexCountMatches<PUML::TETRAHEDRON>());
static_assert(faceVertexCountMatches<PUML::TRIANGLE>());
static_assert(faceVertexCountMatches<PUML::QUADRANGLE>());

static_assert(Topology<PUML::TETRAHEDRON>::faceedges() ==
              Topology<PUML::TETRAHEDRON>::facevertices());

/// Every face is described by vertices of the cell it belongs to.
template <PUML::TopoType Topo>
void checkFaceTable() {
  const auto* faces = Numbering<Topo>::facevertices();

  for (unsigned int face = 0; face < Topology<Topo>::cellfaces(); ++face) {
    for (unsigned int v = 0; v < Topology<Topo>::facevertices(); ++v) {
      EXPECT_LT(faces[face][v], Topology<Topo>::cellvertices()) << "face " << face;
    }
  }
}

TEST(Topology, FaceTablesReferenceCellVertices) {
  checkFaceTable<PUML::TETRAHEDRON>();
  checkFaceTable<PUML::TRIANGLE>();
  checkFaceTable<PUML::QUADRANGLE>();
}

TEST(Topology, EdgeTableReferencesCellVertices) {
  const auto* edges = Numbering<PUML::TETRAHEDRON>::edgevertices();

  for (unsigned int edge = 0; edge < Topology<PUML::TETRAHEDRON>::celledges(); ++edge) {
    EXPECT_LT(edges[edge][0], Topology<PUML::TETRAHEDRON>::cellvertices());
    EXPECT_LT(edges[edge][1], Topology<PUML::TETRAHEDRON>::cellvertices());
    EXPECT_NE(edges[edge][0], edges[edge][1]) << "edge " << edge;
  }
}

/// The two faces listed for an edge have to be the two faces that contain both
/// of its vertices.
TEST(Topology, EdgeAdjacencyAgreesWithTheFaceTable) {
  using Topo = Topology<PUML::TETRAHEDRON>;
  const auto* edges = Numbering<PUML::TETRAHEDRON>::edgevertices();
  const auto* edgeFaces = Numbering<PUML::TETRAHEDRON>::edgefaces();
  const auto* faces = Numbering<PUML::TETRAHEDRON>::facevertices();

  const auto faceContains = [&](unsigned int face, unsigned int vertex) {
    for (unsigned int v = 0; v < Topo::facevertices(); ++v) {
      if (faces[face][v] == vertex) {
        return true;
      }
    }
    return false;
  };

  for (unsigned int edge = 0; edge < Topo::celledges(); ++edge) {
    unsigned int found = 0;
    for (unsigned int face = 0; face < Topo::cellfaces(); ++face) {
      if (faceContains(face, edges[edge][0]) && faceContains(face, edges[edge][1])) {
        ++found;
        EXPECT_TRUE(edgeFaces[edge][0] == face || edgeFaces[edge][1] == face)
            << "edge " << edge << " is missing face " << face;
      }
    }
    EXPECT_EQ(found, 2U) << "edge " << edge;
    EXPECT_NE(edgeFaces[edge][0], edgeFaces[edge][1]) << "edge " << edge;
  }
}

} // namespace
