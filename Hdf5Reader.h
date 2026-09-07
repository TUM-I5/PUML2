// SPDX-FileCopyrightText: 2026 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file
 *  This file is part of PUML
 *
 *  For conditions of distribution and use, please see the copyright
 *  notice in the file 'COPYING' at the root directory of this package
 *  and the copyright notice at https://github.com/TUM-I5/PUML
 */

#ifndef PUML_HDF5READER_H
#define PUML_HDF5READER_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include <hdf5.h>

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include "utils/logger.h"
#include "utils/stringutils.h"

#include "CellType.h"
#include "PUML.h"
#include "Topology.h"

namespace PUML {

template <typename T>
class HDF5TypeInfer {
  public:
  static auto type() -> hid_t { return -1; }
};
template <>
class HDF5TypeInfer<char> {
  public:
  static auto type() -> hid_t { return H5T_NATIVE_CHAR; }
};
template <>
class HDF5TypeInfer<signed char> {
  public:
  static auto type() -> hid_t { return H5T_NATIVE_SCHAR; }
};
template <>
class HDF5TypeInfer<unsigned char> {
  public:
  static auto type() -> hid_t { return H5T_NATIVE_UCHAR; }
};
template <>
class HDF5TypeInfer<short> {
  public:
  static auto type() -> hid_t { return H5T_NATIVE_SHORT; }
};
template <>
class HDF5TypeInfer<unsigned short> {
  public:
  static auto type() -> hid_t { return H5T_NATIVE_USHORT; }
};
template <>
class HDF5TypeInfer<int> {
  public:
  static auto type() -> hid_t { return H5T_NATIVE_INT; }
};
template <>
class HDF5TypeInfer<unsigned> {
  public:
  static auto type() -> hid_t { return H5T_NATIVE_UINT; }
};
template <>
class HDF5TypeInfer<long> {
  public:
  static auto type() -> hid_t { return H5T_NATIVE_LONG; }
};
template <>
class HDF5TypeInfer<unsigned long> {
  public:
  static auto type() -> hid_t { return H5T_NATIVE_ULONG; }
};
template <>
class HDF5TypeInfer<long long> {
  public:
  static auto type() -> hid_t { return H5T_NATIVE_LLONG; }
};
template <>
class HDF5TypeInfer<unsigned long long> {
  public:
  static auto type() -> hid_t { return H5T_NATIVE_ULLONG; }
};
template <>
class HDF5TypeInfer<float> {
  public:
  static auto type() -> hid_t { return H5T_NATIVE_FLOAT; }
};
template <>
class HDF5TypeInfer<double> {
  public:
  static auto type() -> hid_t { return H5T_NATIVE_DOUBLE; }
};
template <>
class HDF5TypeInfer<long double> {
  public:
  static auto type() -> hid_t { return H5T_NATIVE_LDOUBLE; }
};

#define checkH5Err(...) checkH5ErrImpl(__VA_ARGS__, __FILE__, __LINE__, m_rank)

/**
 * Fills a mesh from HDF5 datasets.
 *
 * A dataset is named "filename:/dataset". The reader spreads the entities of a
 * dataset evenly over the ranks, which is the split PUML::setTotalSize
 * describes.
 */
template <TopoType Topo>
class Hdf5Reader {
  public:
  /**
   * @param puml The mesh to fill
   * @param defaultType The kind of cell to assume for a file that does not say,
   *        which is what a file of one kind of cell looks like
   */
  explicit Hdf5Reader(PUML<Topo>& puml, CellType defaultType = CellType::Tetrahedron)
      : m_puml(puml), m_defaultType(defaultType) {
#ifdef USE_MPI
    m_comm = puml.comm();
    MPI_Comm_rank(m_comm, &m_rank);
#endif // USE_MPI
  }

  Hdf5Reader(const Hdf5Reader&) = delete;
  auto operator=(const Hdf5Reader&) -> Hdf5Reader& = delete;
  Hdf5Reader(Hdf5Reader&&) = delete;
  auto operator=(Hdf5Reader&&) -> Hdf5Reader& = delete;

  ~Hdf5Reader() { closeFile(); }

  /**
   * Releases the file the reader is holding open.
   */
  void close() { closeFile(); }

  /**
   * Reads a cell and a vertex dataset. Equivalent to inferSize() followed by
   * addData() for each of the two.
   */
  void open(const std::string& cellName, const std::string& vertexName) {
    const auto cellNames = utils::StringUtils::split(cellName, ':');
    if (cellNames.size() != 2) {
      logError() << "Cells name must have the form \"filename:/dataset\"";
    }

    const auto vertexNames = utils::StringUtils::split(vertexName, ':');
    if (vertexNames.size() != 2) {
      logError() << "Vertices name must have the form \"filename:/dataset\"";
    }

    // infer sizes from the data
    inferSize(DataType::Cell, cellName);
    inferSize(DataType::Vertex, vertexName);

    logInfo() << "Found" << m_puml.distributor(DataType::Cell).totalSize() << "cells";
    logInfo() << "Found" << m_puml.distributor(DataType::Vertex).totalSize() << "vertices";

    // now actually read the data
    addData<unsigned long>(
        "connectivity", cellName, DataType::Cell, {internal::Topology<Topo>::cellvertices()});
    addData<double>("geometry", vertexName, DataType::Vertex, {3});
  }

  void inferSize(DataType type, const std::string& dataset) {
    const hid_t h5file = fileOf(dataset);

    // Get cell dataset
    hid_t h5dataset = H5Dopen(h5file, datasetOf(dataset).c_str(), H5P_DEFAULT);
    checkH5Err(h5dataset);

    // Check the size of cell dataset
    hid_t h5space = H5Dget_space(h5dataset);
    const auto ndims = H5Sget_simple_extent_ndims(h5space);
    checkH5Err(h5space);
    if (H5Sget_simple_extent_ndims(h5space) < 1) {
      logError() << "Size inference dataset must have at least one dimension";
    }
    std::vector<hsize_t> dims(ndims);
    checkH5Err(H5Sget_simple_extent_dims(h5space, dims.data(), nullptr));

    m_puml.setTotalSize(type, dims[0]);

    checkH5Err(H5Sclose(h5space));
    checkH5Err(H5Dclose(h5dataset));
  }

  template <typename T>
  auto addData(const std::string& path,
               DataType type,
               const std::vector<size_t>& sizes
#ifdef USE_MPI
               ,
               MPI_Datatype mpiType = MPITypeInfer<T>::type()
#endif
                   ,
               hid_t hdf5Type = HDF5TypeInfer<T>::type()) -> int {
    const auto [name, ret] = m_puml.nextLegacyName(type);

    addData<T>(name,
               path,
               type,
               sizes
#ifdef USE_MPI
               ,
               mpiType
#endif
               ,
               hdf5Type);

    return ret;
  }

  template <typename T = int>
  void addData(const std::string& name,
               const std::string& path,
               DataType type,
               const std::vector<size_t>& sizes
#ifdef USE_MPI
               ,
               MPI_Datatype mpiType = MPITypeInfer<T>::type()
#endif
                   ,
               hid_t hdf5Type = HDF5TypeInfer<T>::type()) {
    static_assert(std::is_trivially_copyable_v<T>, "T needs to be trivially copyable");
    static_assert(std::is_trivially_default_constructible_v<T>,
                  "T needs to be trivially default constructible");
    const auto& cellDistributor = m_puml.distributor(type);
    const hid_t h5file = fileOf(path);
    const auto dataName = datasetOf(path);

    const unsigned long totalSize = cellDistributor.totalSize();

    // Get cell dataset
    hid_t h5dataset = H5Dopen(h5file, dataName.c_str(), H5P_DEFAULT);
    checkH5Err(h5dataset);

    // Check the size of cell dataset
    hid_t h5space = H5Dget_space(h5dataset);
    checkH5Err(h5space);
    const auto dimcount = H5Sget_simple_extent_ndims(h5space);
    if (dimcount != static_cast<int>(1 + sizes.size())) {
      logError() << "Dataset must have" << 1 + sizes.size() << "dimension(s), but it has"
                 << dimcount;
    }
    std::vector<hsize_t> dim(1 + sizes.size());
    checkH5Err(H5Sget_simple_extent_dims(h5space, dim.data(), nullptr));
    if (dim[0] != totalSize) {
      logError() << "Dataset has the wrong size:" << dim[0] << "vs." << totalSize;
    }
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      if (dim[i + 1] != sizes[i]) {
        const std::vector<hsize_t> subdims(dim.begin() + 1, dim.end());
        logError() << "Dataset has the wrong subsize:" << subdims << "vs." << sizes;
      }
    }

    // Read the cells
    auto [offset, localSize] = cellDistributor.offsetAndSize(m_rank);

    size_t elemSize = 1;
    for (auto size : sizes) {
      elemSize *= size;
    }

    std::vector<hsize_t> start = {offset};
    std::vector<hsize_t> count = {localSize};

    for (auto size : sizes) {
      start.push_back(0);
      count.push_back(size);
    }

    checkH5Err(
        H5Sselect_hyperslab(h5space, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr));

    hid_t h5memspace = H5Screate_simple(static_cast<int>(count.size()), count.data(), nullptr);
    checkH5Err(h5memspace);

    hid_t h5alist = H5Pcreate(H5P_DATASET_XFER);
    checkH5Err(h5alist);
#ifdef USE_MPI
    checkH5Err(H5Pset_dxpl_mpio(h5alist, H5FD_MPIO_COLLECTIVE));
#endif // USE_MPI

    const auto handle = m_puml.template allocateData<T>(name,
                                                        type,
                                                        sizes
#ifdef USE_MPI
                                                        ,
                                                        mpiType
#endif // USE_MPI
    );
    checkH5Err(
        H5Dread(h5dataset, hdf5Type, h5memspace, h5space, h5alist, m_puml.data(handle).data()));

    // Close data
    checkH5Err(H5Sclose(h5space));
    checkH5Err(H5Sclose(h5memspace));
    checkH5Err(H5Dclose(h5dataset));
    checkH5Err(H5Pclose(h5alist));
  }

  /**
   * Reads a mesh whose cells need not all be of the same kind.
   *
   * The connectivity is one flat array in which every cell is preceded by the
   * kind it is of, and the offsets say where each cell begins in it. A file
   * without the offsets holds cells of one kind only, laid out as a rectangle,
   * and is read as the kind this reader was given.
   *
   * @param connectivityName The flat connectivity, as "file:/dataset"
   * @param offsetsName The offsets into it; may be absent from the file
   * @param vertexName The vertex positions
   */
  void openMixed(const std::string& connectivityName,
                 const std::string& offsetsName,
                 const std::string& vertexName) {
    static_assert(Topo == MIXED, "only a mixed mesh is read this way");

    if (exists(offsetsName)) {
      readRaggedCells(connectivityName, offsetsName);
    } else {
      readUniformCells(connectivityName);
    }

    inferSize(DataType::Vertex, vertexName);
    addData<double>("geometry", vertexName, DataType::Vertex, {3});

    logInfo() << "Found" << m_puml.distributor(DataType::Cell).totalSize() << "cells";
    logInfo() << "Found" << m_puml.distributor(DataType::Vertex).totalSize() << "vertices";
  }

  /**
   * Whether the file holds the given dataset.
   */
  auto exists(const std::string& path) -> bool {
    const hid_t h5file = fileOf(path);
    return H5Lexists(h5file, datasetOf(path).c_str(), H5P_DEFAULT) > 0;
  }

  private:
  /**
   * Reads a flat connectivity, splitting every cell into the kind it is of and
   * the vertices it is built from.
   */
  void readRaggedCells(const std::string& connectivityName, const std::string& offsetsName) {
    // The offsets hold one entry per cell plus one behind the last.
    const auto totalCells = datasetLength(offsetsName) - 1;
    m_puml.setTotalSize(DataType::Cell, totalCells);

    const auto [firstCell, cellCount] = m_puml.distributor(DataType::Cell).offsetAndSize(m_rank);

    std::vector<GlobalId> offsets(cellCount + 1);
    readSlab(offsetsName, firstCell, cellCount + 1, offsets.data());

    const auto base = offsets.front();
    std::vector<GlobalId> flat(offsets.back() - base);
    readSlab(connectivityName, base, flat.size(), flat.data());

    const auto handle = m_puml.template allocateData<GlobalId>(
        "connectivity", DataType::Cell, {internal::MaxCellVertices});
    auto view = m_puml.data(handle);
    std::vector<CellType> types(cellCount);

    for (Size i = 0; i < cellCount; ++i) {
      const auto first = offsets[i] - base;
      const auto last = offsets[i + 1] - base;
      if (last <= first) {
        throwError("cell", firstCell + i, "holds no values at all");
      }

      const auto type = static_cast<CellType>(flat[first]);
      if (!internal::isSupported(type)) {
        throwError("cell",
                   firstCell + i,
                   "is of kind",
                   static_cast<unsigned int>(flat[first]),
                   "which a mesh cannot be built from");
      }
      const auto& shape = internal::shapeOf(type);
      if (last - first - 1 != shape.vertexCount) {
        throwError("cell",
                   firstCell + i,
                   "is a",
                   internal::nameOf(type),
                   "and needs",
                   shape.vertexCount,
                   "vertices, but the file gives it",
                   last - first - 1);
      }

      types[i] = type;
      auto* cell = view.entity(i);
      for (Size v = 0; v < internal::MaxCellVertices; ++v) {
        cell[v] = flat[first + 1 + std::min<Size>(v, shape.vertexCount - 1)];
      }
    }

    m_puml.setCellTypes(types.data());
  }

  /**
   * Reads a rectangular connectivity of one kind of cell into a mixed mesh.
   */
  void readUniformCells(const std::string& connectivityName) {
    const auto& shape = internal::shapeOf(m_defaultType);

    inferSize(DataType::Cell, connectivityName);
    const auto cellCount = m_puml.numOriginalCells();

    std::vector<GlobalId> raw(cellCount * shape.vertexCount);
    const auto firstCell = m_puml.distributor(DataType::Cell).offsetAndSize(m_rank).first;
    readSlab2d(connectivityName, firstCell, cellCount, shape.vertexCount, raw.data());

    const auto handle = m_puml.template allocateData<GlobalId>(
        "connectivity", DataType::Cell, {internal::MaxCellVertices});
    auto view = m_puml.data(handle);
    for (Size i = 0; i < cellCount; ++i) {
      auto* cell = view.entity(i);
      for (Size v = 0; v < internal::MaxCellVertices; ++v) {
        cell[v] = raw[(i * shape.vertexCount) + std::min<Size>(v, shape.vertexCount - 1)];
      }
    }

    const std::vector<CellType> types(cellCount, m_defaultType);
    m_puml.setCellTypes(types.data());
  }

  /// The length of the first dimension of a dataset.
  auto datasetLength(const std::string& path) -> hsize_t {
    const hid_t h5file = fileOf(path);
    hid_t h5dataset = H5Dopen(h5file, datasetOf(path).c_str(), H5P_DEFAULT);
    checkH5Err(h5dataset);
    hid_t h5space = H5Dget_space(h5dataset);
    checkH5Err(h5space);
    std::vector<hsize_t> dim(H5Sget_simple_extent_ndims(h5space));
    checkH5Err(H5Sget_simple_extent_dims(h5space, dim.data(), nullptr));
    checkH5Err(H5Sclose(h5space));
    checkH5Err(H5Dclose(h5dataset));
    return dim[0];
  }

  /// Reads count values of a one-dimensional dataset, starting at offset.
  void readSlab(const std::string& path, hsize_t offset, hsize_t count, GlobalId* into) {
    readSlabImpl(path, {offset, 0}, {count, 0}, 1, into);
  }

  /// Reads count rows of a two-dimensional dataset, starting at offset.
  void readSlab2d(
      const std::string& path, hsize_t offset, hsize_t count, hsize_t width, GlobalId* into) {
    readSlabImpl(path, {offset, 0}, {count, width}, 2, into);
  }

  void readSlabImpl(const std::string& path,
                    std::array<hsize_t, 2> start,
                    std::array<hsize_t, 2> count,
                    int rank,
                    GlobalId* into) {
    const hid_t h5file = fileOf(path);
    hid_t h5dataset = H5Dopen(h5file, datasetOf(path).c_str(), H5P_DEFAULT);
    checkH5Err(h5dataset);

    hid_t h5space = H5Dget_space(h5dataset);
    checkH5Err(h5space);
    checkH5Err(
        H5Sselect_hyperslab(h5space, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr));

    hid_t h5memspace = H5Screate_simple(rank, count.data(), nullptr);
    checkH5Err(h5memspace);

    hid_t h5alist = H5Pcreate(H5P_DATASET_XFER);
    checkH5Err(h5alist);
#ifdef USE_MPI
    checkH5Err(H5Pset_dxpl_mpio(h5alist, H5FD_MPIO_COLLECTIVE));
#endif // USE_MPI

    checkH5Err(
        H5Dread(h5dataset, HDF5TypeInfer<GlobalId>::type(), h5memspace, h5space, h5alist, into));

    checkH5Err(H5Pclose(h5alist));
    checkH5Err(H5Sclose(h5memspace));
    checkH5Err(H5Sclose(h5space));
    checkH5Err(H5Dclose(h5dataset));
  }

  /**
   * The file a dataset lives in, opened at the first dataset and kept open for
   * the ones that follow.
   */
  auto fileOf(const std::string& path) -> hid_t {
    const auto names = utils::StringUtils::split(path, ':');
    if (names.size() != 2) {
      logError() << "Invalid dataset name" << path << "; expected the form file:/dataset";
    }

    if (m_file >= 0 && names[0] == m_fileName) {
      return m_file;
    }
    closeFile();

    hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
    checkH5Err(plist);
#ifdef USE_MPI
    checkH5Err(H5Pset_fapl_mpio(plist, m_comm, MPI_INFO_NULL));
#endif // USE_MPI

    m_file = H5Fopen(names[0].c_str(), H5F_ACC_RDONLY, plist);
    checkH5Err(m_file);
    checkH5Err(H5Pclose(plist));

    m_fileName = names[0];
    return m_file;
  }

  /// The dataset part of a "file:/dataset" name.
  static auto datasetOf(const std::string& path) -> std::string {
    return utils::StringUtils::split(path, ':')[1];
  }

  void closeFile() {
    if (m_file >= 0) {
      H5Fclose(m_file);
      m_file = -1;
      m_fileName.clear();
    }
  }

  template <typename TT>
  static void checkH5ErrImpl(TT status, const char* file, int line, int rank) {
    if (status < 0) {
      logError() << utils::nospace << "An HDF5 error occurred in PUML (" << file << ": " << line
                 << ") on rank " << rank;
    }
  }

  PUML<Topo>& m_puml;
  CellType m_defaultType{CellType::Tetrahedron};
  hid_t m_file{-1};
  std::string m_fileName;
  int m_rank{0};
#ifdef USE_MPI
  MPI_Comm m_comm{MPI_COMM_WORLD};
#endif // USE_MPI
};

#undef checkH5Err

} // namespace PUML

#endif // PUML_HDF5READER_H
