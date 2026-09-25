// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PUML_TEST_PUMLTEST_H
#define PUML_TEST_PUMLTEST_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include "CellType.h"
#include "PUML.h"
#include "Upward.h"

namespace puml::test {

inline auto commRank() -> int {
  int rank = 0;
#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif // USE_MPI
  return rank;
}

inline auto commSize() -> int {
  int procs = 1;
#ifdef USE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
#endif // USE_MPI
  return procs;
}

inline auto globalSum(long value) -> long {
#ifdef USE_MPI
  long total = 0;
  MPI_Allreduce(&value, &total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  return total;
#else  // USE_MPI
  return value;
#endif // USE_MPI
}

/// Collects the given per-rank values of all ranks into one array.
inline auto gather(const std::vector<unsigned long>& local) -> std::vector<unsigned long> {
#ifdef USE_MPI
  const int procs = commSize();

  std::vector<int> counts(procs);
  int localCount = static_cast<int>(local.size());
  MPI_Allgather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> displs(procs);
  for (int i = 1; i < procs; ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }

  std::vector<unsigned long> all(displs.back() + counts.back());
  MPI_Allgatherv(local.data(),
                 localCount,
                 MPI_UNSIGNED_LONG,
                 all.data(),
                 counts.data(),
                 displs.data(),
                 MPI_UNSIGNED_LONG,
                 MPI_COMM_WORLD);
  return all;
#else  // USE_MPI
  return local;
#endif // USE_MPI
}

/// An entity is owned by the lowest rank that holds it, which is the rule PUML
/// follows when it hands out global ids.
template <typename T>
auto ownedGids(const std::vector<T>& elements) -> std::vector<unsigned long> {
  const int rank = commRank();

  std::vector<unsigned long> gids;
  for (const auto& element : elements) {
    if (element.shared().empty() || element.shared()[0] > rank) {
      gids.push_back(element.gid());
    }
  }
  return gids;
}

/// Collects the global ids of all entities of one kind, over all ranks.
template <typename T>
auto allGids(const std::vector<T>& elements) -> std::vector<unsigned long> {
  auto all = gather(ownedGids(elements));
  std::sort(all.begin(), all.end());
  return all;
}

/// True if the values are exactly 0, 1, ..., n-1.
inline auto isContiguousFromZero(const std::vector<unsigned long>& sorted) -> bool {
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    if (sorted[i] != i) {
      return false;
    }
  }
  return true;
}

struct Split {
  std::size_t offset{0};
  std::size_t size{0};
};

/// The split the HDF5 reader produces.
inline auto evenSplit(std::size_t total, int rank, int procs) -> Split {
  const std::size_t perRank = total / procs;
  const std::size_t rest = total % procs;
  if (static_cast<std::size_t>(rank) < rest) {
    return {rank * (perRank + 1), perRank + 1};
  }
  return {rest * (perRank + 1) + (rank - rest) * perRank, perRank};
}

/// A split in which the first rank holds a single entity and the remaining
/// ranks share what is left.
inline auto lopsidedSplit(std::size_t total, int rank, int procs) -> Split {
  if (procs == 1) {
    return {0, total};
  }
  if (rank == 0) {
    return {0, 1};
  }
  const std::size_t rest = (total - 1) / (procs - 1);
  const std::size_t offset = 1 + (rank - 1) * rest;
  return {offset, (rank == procs - 1) ? total - offset : rest};
}

struct CubeMesh {
  std::vector<unsigned long> connect; // numCells * 4
  std::vector<double> geometry;       // numVertices * 3
  std::size_t numCells{0};
  std::size_t numVertices{0};
  int n{0};

  /// Faces on the surface of the cube, each hexahedron face split in two.
  [[nodiscard]] auto numBoundaryFaces() const -> long { return 4L * 3L * n * n; }
};

/// Kuhn subdivision of every hexahedron of an n x n x n grid into six
/// tetrahedra. All hexahedra use the same local diagonal, so the subdivision is
/// conforming across hexahedron faces.
inline auto makeCubeMesh(int n) -> CubeMesh {
  const auto vid = [&](int i, int j, int k) {
    return static_cast<unsigned long>((k * (n + 1) + j) * (n + 1) + i);
  };

  CubeMesh mesh;
  mesh.n = n;
  mesh.numVertices = static_cast<std::size_t>(n + 1) * (n + 1) * (n + 1);
  mesh.geometry.resize(mesh.numVertices * 3);
  for (int k = 0; k <= n; ++k) {
    for (int j = 0; j <= n; ++j) {
      for (int i = 0; i <= n; ++i) {
        const auto v = vid(i, j, k);
        mesh.geometry[3 * v + 0] = i;
        mesh.geometry[3 * v + 1] = j;
        mesh.geometry[3 * v + 2] = k;
      }
    }
  }

  static const int Tets[6][4] = {
      {0, 1, 3, 7}, {0, 1, 5, 7}, {0, 4, 5, 7}, {0, 2, 3, 7}, {0, 2, 6, 7}, {0, 4, 6, 7}};

  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        std::array<unsigned long, 8> corner{};
        for (int b = 0; b < 8; ++b) {
          corner[b] = vid(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1));
        }
        for (const auto& tet : Tets) {
          for (const int c : tet) {
            mesh.connect.push_back(corner[c]);
          }
        }
      }
    }
  }
  mesh.numCells = mesh.connect.size() / 4;
  return mesh;
}

struct HexCubeMesh {
  std::vector<unsigned long> connect; // numCells * 8
  std::vector<double> geometry;       // numVertices * 3
  std::size_t numCells{0};
  std::size_t numVertices{0};
  int n{0};

  [[nodiscard]] auto numBoundaryFaces() const -> long { return 6L * n * n; }
};

/// An n x n x n grid of hexahedra, numbered the way XDMF numbers them: the four
/// vertices of the lower face counter-clockwise, then the four above them.
inline auto makeHexCubeMesh(int n) -> HexCubeMesh {
  const auto vid = [&](int i, int j, int k) {
    return static_cast<unsigned long>((k * (n + 1) + j) * (n + 1) + i);
  };

  HexCubeMesh mesh;
  mesh.n = n;
  mesh.numVertices = static_cast<std::size_t>(n + 1) * (n + 1) * (n + 1);
  mesh.geometry.resize(mesh.numVertices * 3);
  for (int k = 0; k <= n; ++k) {
    for (int j = 0; j <= n; ++j) {
      for (int i = 0; i <= n; ++i) {
        const auto v = vid(i, j, k);
        mesh.geometry[3 * v + 0] = i;
        mesh.geometry[3 * v + 1] = j;
        mesh.geometry[3 * v + 2] = k;
      }
    }
  }

  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        for (const auto& c : {std::array<int, 3>{0, 0, 0},
                              {1, 0, 0},
                              {1, 1, 0},
                              {0, 1, 0},
                              {0, 0, 1},
                              {1, 0, 1},
                              {1, 1, 1},
                              {0, 1, 1}}) {
          mesh.connect.push_back(vid(i + c[0], j + c[1], k + c[2]));
        }
      }
    }
  }
  mesh.numCells = mesh.connect.size() / 8;
  return mesh;
}

inline void feed(PUML::HEXPUML& puml, const HexCubeMesh& mesh, Split cells, Split vertices) {
#ifdef USE_MPI
  puml.setComm(MPI_COMM_WORLD);
#endif // USE_MPI

  puml.setSize(PUML::CELL, cells.size);
  puml.setSize(PUML::VERTEX, vertices.size);
  puml.setConnectivity(puml.addDataArray<unsigned long>(
      "connectivity", mesh.connect.data() + 8 * cells.offset, PUML::CELL, {8}));
  puml.setGeometry(puml.addDataArray<double>(
      "geometry", mesh.geometry.data() + 3 * vertices.offset, PUML::VERTEX, {3}));
}

struct SquareMesh {
  std::vector<unsigned long> connect; // numCells * 3
  std::vector<double> geometry;       // numVertices * 3
  std::size_t numCells{0};
  std::size_t numVertices{0};
  int n{0};

  /// The mesh edges on the border of the square.
  [[nodiscard]] auto numBoundaryFaces() const -> long { return 4L * n; }

  /// Every square contributes two triangles, and the grid lines and diagonals
  /// give the mesh edges.
  [[nodiscard]] auto numFaces() const -> long {
    return 2L * n * (n + 1) + static_cast<long>(n) * n;
  }
};

/// An n x n grid of squares, each split along one diagonal. The mesh is two
/// dimensional but sits in space, so its vertices carry three coordinates.
inline auto makeSquareMesh(int n) -> SquareMesh {
  const auto vid = [&](int i, int j) { return static_cast<unsigned long>(j * (n + 1) + i); };

  SquareMesh mesh;
  mesh.n = n;
  mesh.numVertices = static_cast<std::size_t>(n + 1) * (n + 1);
  mesh.geometry.resize(mesh.numVertices * 3);
  for (int j = 0; j <= n; ++j) {
    for (int i = 0; i <= n; ++i) {
      const auto v = vid(i, j);
      mesh.geometry[3 * v + 0] = i;
      mesh.geometry[3 * v + 1] = j;
      mesh.geometry[3 * v + 2] = 0.0;
    }
  }

  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      for (const auto& tri : {std::array<int, 6>{0, 0, 1, 0, 1, 1}, {0, 0, 1, 1, 0, 1}}) {
        for (int c = 0; c < 3; ++c) {
          mesh.connect.push_back(vid(i + tri[2 * c], j + tri[(2 * c) + 1]));
        }
      }
    }
  }
  mesh.numCells = mesh.connect.size() / 3;
  return mesh;
}

inline void
    feed(PUML::PUML<PUML::TRIANGLE>& puml, const SquareMesh& mesh, Split cells, Split vertices) {
#ifdef USE_MPI
  puml.setComm(MPI_COMM_WORLD);
#endif // USE_MPI

  puml.setSize(PUML::CELL, cells.size);
  puml.setSize(PUML::VERTEX, vertices.size);
  puml.setConnectivity(puml.addDataArray<unsigned long>(
      "connectivity", mesh.connect.data() + 3 * cells.offset, PUML::CELL, {3}));
  puml.setGeometry(puml.addDataArray<double>(
      "geometry", mesh.geometry.data() + 3 * vertices.offset, PUML::VERTEX, {3}));
}

/// A mesh of both hexahedra and tetrahedra, in two blocks that do not touch:
/// a hexahedral grid and a Kuhn-subdivided one, set apart along x. A shared
/// face between the two would need a pyramid or a wedge, which comes later.
struct MixedMesh {
  std::vector<unsigned long> connect; // padded to eight vertices per cell
  std::vector<std::uint8_t> types;
  std::vector<double> geometry;
  std::size_t numCells{0};
  std::size_t numVertices{0};
  std::size_t numHexCells{0};
  std::size_t numTetCells{0};
  std::size_t numPyramidCells{0};
  std::size_t numWedgeCells{0};
  long numFaces{0};
  long numEdges{0};
  long numBoundaryFaces{0};
};

inline auto makeMixedMesh(int n) -> MixedMesh {
  const auto hex = makeHexCubeMesh(n);
  const auto tet = makeCubeMesh(n);

  MixedMesh mesh;
  mesh.numHexCells = hex.numCells;
  mesh.numTetCells = tet.numCells;
  mesh.numCells = hex.numCells + tet.numCells;
  mesh.numVertices = hex.numVertices + tet.numVertices;

  mesh.geometry = hex.geometry;
  for (std::size_t v = 0; v < tet.numVertices; ++v) {
    mesh.geometry.push_back(tet.geometry[3 * v + 0] + 2.0 * n);
    mesh.geometry.push_back(tet.geometry[3 * v + 1]);
    mesh.geometry.push_back(tet.geometry[3 * v + 2]);
  }

  const auto pad = [&mesh](const unsigned long* first, std::size_t count) {
    for (std::size_t i = 0; i < 8; ++i) {
      mesh.connect.push_back(i < count ? first[i] : first[0]);
    }
  };

  for (std::size_t c = 0; c < hex.numCells; ++c) {
    pad(hex.connect.data() + 8 * c, 8);
    mesh.types.push_back(static_cast<std::uint8_t>(PUML::CellType::Hexahedron));
  }
  for (std::size_t c = 0; c < tet.numCells; ++c) {
    const auto* first = tet.connect.data() + 4 * c;
    std::array<unsigned long, 4> shifted{};
    for (std::size_t i = 0; i < 4; ++i) {
      shifted[i] = first[i] + hex.numVertices;
    }
    pad(shifted.data(), 4);
    mesh.types.push_back(static_cast<std::uint8_t>(PUML::CellType::Tetrahedron));
  }

  mesh.numBoundaryFaces = hex.numBoundaryFaces() + tet.numBoundaryFaces();

  // Every hexahedron brings six faces and every tetrahedron four, and an
  // interior face is brought by both of its cells.
  mesh.numFaces = (6L * static_cast<long>(mesh.numHexCells) +
                   4L * static_cast<long>(mesh.numTetCells) + mesh.numBoundaryFaces) /
                  2;

  // Two blocks that do not touch, so the Euler characteristic counts twice.
  mesh.numEdges =
      static_cast<long>(mesh.numVertices) + mesh.numFaces - static_cast<long>(mesh.numCells) - 2;
  return mesh;
}

/// A hexahedral grid where the cells of the first layer are each split into six
/// pyramids with their apex in the middle. The base of every pyramid is a face
/// of the hexahedron it came from, so the mesh stays conforming across the
/// boundary between the two kinds.
inline auto makeHexPyramidMesh(int n) -> MixedMesh {
  const auto hex = makeHexCubeMesh(n);
  const auto& faces = PUML::internal::shapeOf(PUML::CellType::Hexahedron).faceVertices;

  MixedMesh mesh;
  mesh.geometry = hex.geometry;
  mesh.numVertices = hex.numVertices;

  const auto pad = [&mesh](const std::array<unsigned long, 8>& corner, std::size_t count) {
    for (std::size_t i = 0; i < 8; ++i) {
      mesh.connect.push_back(i < count ? corner[i] : corner[0]);
    }
  };

  long splitCells = 0;
  for (std::size_t c = 0; c < hex.numCells; ++c) {
    const auto* corner = hex.connect.data() + 8 * c;
    // The cells of the first layer are the ones with the lowest index.
    const bool split = (c % static_cast<std::size_t>(n)) == 0;

    if (!split) {
      std::array<unsigned long, 8> cell{};
      std::copy(corner, corner + 8, cell.begin());
      pad(cell, 8);
      mesh.types.push_back(static_cast<std::uint8_t>(PUML::CellType::Hexahedron));
      ++mesh.numHexCells;
      continue;
    }

    // One vertex in the middle of the hexahedron, shared by its six pyramids.
    const auto apex = static_cast<unsigned long>(mesh.numVertices);
    ++mesh.numVertices;
    for (std::size_t d = 0; d < 3; ++d) {
      double sum = 0.0;
      for (std::size_t v = 0; v < 8; ++v) {
        sum += hex.geometry[3 * corner[v] + d];
      }
      mesh.geometry.push_back(sum / 8.0);
    }

    for (std::size_t f = 0; f < 6; ++f) {
      std::array<unsigned long, 8> cell{};
      for (std::size_t v = 0; v < 4; ++v) {
        cell[v] = corner[faces[f][v]];
      }
      cell[4] = apex;
      pad(cell, 5);
      mesh.types.push_back(static_cast<std::uint8_t>(PUML::CellType::Pyramid));
    }
    mesh.numPyramidCells += 6;
    ++splitCells;
  }

  mesh.numCells = mesh.types.size();
  mesh.numBoundaryFaces = hex.numBoundaryFaces();

  // Six faces per hexahedron and five per pyramid, an interior face brought by
  // both of its cells.
  mesh.numFaces = (6L * static_cast<long>(mesh.numHexCells) +
                   5L * static_cast<long>(mesh.numPyramidCells) + mesh.numBoundaryFaces) /
                  2;
  // One connected body.
  mesh.numEdges =
      static_cast<long>(mesh.numVertices) + mesh.numFaces - static_cast<long>(mesh.numCells) - 1;
  static_cast<void>(splitCells);
  return mesh;
}

/// A triangular mesh extruded into wedges, layer by layer.
inline auto makeWedgeMesh(int n) -> MixedMesh {
  const auto flat = makeSquareMesh(n);

  MixedMesh mesh;
  mesh.numVertices = flat.numVertices * static_cast<std::size_t>(n + 1);
  for (int layer = 0; layer <= n; ++layer) {
    for (std::size_t v = 0; v < flat.numVertices; ++v) {
      mesh.geometry.push_back(flat.geometry[3 * v + 0]);
      mesh.geometry.push_back(flat.geometry[3 * v + 1]);
      mesh.geometry.push_back(layer);
    }
  }

  for (int layer = 0; layer < n; ++layer) {
    const auto below = static_cast<unsigned long>(layer) * flat.numVertices;
    const auto above = below + flat.numVertices;
    for (std::size_t c = 0; c < flat.numCells; ++c) {
      const auto* tri = flat.connect.data() + 3 * c;
      for (std::size_t v = 0; v < 3; ++v) {
        mesh.connect.push_back(tri[v] + below);
      }
      for (std::size_t v = 0; v < 3; ++v) {
        mesh.connect.push_back(tri[v] + above);
      }
      mesh.connect.push_back(tri[0] + below);
      mesh.connect.push_back(tri[0] + below);
      mesh.types.push_back(static_cast<std::uint8_t>(PUML::CellType::Wedge));
    }
  }

  mesh.numWedgeCells = mesh.types.size();
  mesh.numCells = mesh.types.size();

  // The triangles of the top and bottom layer, plus the sides of the column.
  mesh.numBoundaryFaces = 2L * static_cast<long>(flat.numCells) + flat.numBoundaryFaces() * n;
  mesh.numFaces = (5L * static_cast<long>(mesh.numCells) + mesh.numBoundaryFaces) / 2;
  mesh.numEdges =
      static_cast<long>(mesh.numVertices) + mesh.numFaces - static_cast<long>(mesh.numCells) - 1;
  return mesh;
}

inline void feed(PUML::MIXEDPUML& puml, const MixedMesh& mesh, Split cells, Split vertices) {
#ifdef USE_MPI
  puml.setComm(MPI_COMM_WORLD);
#endif // USE_MPI

  puml.setSize(PUML::CELL, cells.size);
  puml.setSize(PUML::VERTEX, vertices.size);
  puml.setConnectivity(puml.addDataArray<unsigned long>(
      "connectivity", mesh.connect.data() + 8 * cells.offset, PUML::CELL, {8}));
  puml.setCellTypes(reinterpret_cast<const PUML::CellType*>(mesh.types.data() + cells.offset));
  puml.setGeometry(puml.addDataArray<double>(
      "geometry", mesh.geometry.data() + 3 * vertices.offset, PUML::VERTEX, {3}));
}

/// Hands the given portion of a cube mesh to PUML without touching a file.
inline void feed(PUML::TETPUML& puml, const CubeMesh& mesh, Split cells, Split vertices) {
#ifdef USE_MPI
  puml.setComm(MPI_COMM_WORLD);
#endif // USE_MPI

  puml.setSize(PUML::CELL, cells.size);
  puml.setSize(PUML::VERTEX, vertices.size);
  puml.setConnectivity(puml.addDataArray<unsigned long>(
      "connectivity", mesh.connect.data() + 4 * cells.offset, PUML::CELL, {4}));
  puml.setGeometry(puml.addDataArray<double>(
      "geometry", mesh.geometry.data() + 3 * vertices.offset, PUML::VERTEX, {3}));
}

struct MeshCounts {
  long cells{0};
  long faces{0};
  long edges{0};
  long vertices{0};
  long boundaryFaces{0};
  long unusedFaces{0};

  [[nodiscard]] auto euler() const -> long { return vertices - edges + faces - cells; }
};

template <PUML::TopoType Topo>
auto measure(const PUML::PUML<Topo>& puml) -> MeshCounts {
  MeshCounts counts;
  counts.cells = globalSum(static_cast<long>(puml.cells().size()));
  counts.faces = globalSum(static_cast<long>(ownedGids(puml.faces()).size()));
  counts.edges = globalSum(static_cast<long>(ownedGids(puml.edges()).size()));
  counts.vertices = globalSum(static_cast<long>(ownedGids(puml.vertices()).size()));

  long boundary = 0;
  long unused = 0;
  for (const auto& face : puml.faces()) {
    std::array<PUML::LocalId, 2> adjacent{};
    PUML::Upward::cells(puml, face, adjacent.data());
    if (adjacent[0] == PUML::InvalidLocalId) {
      ++unused;
    } else if (adjacent[1] == PUML::InvalidLocalId && face.shared().empty()) {
      ++boundary;
    }
  }
  counts.boundaryFaces = globalSum(boundary);
  counts.unusedFaces = globalSum(unused);
  return counts;
}

} // namespace puml::test

#endif // PUML_TEST_PUMLTEST_H
