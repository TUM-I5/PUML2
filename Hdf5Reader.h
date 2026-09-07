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

#include <cstddef>
#include <string>
#include <vector>

#include <hdf5.h>

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include "utils/logger.h"
#include "utils/stringutils.h"

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
  explicit Hdf5Reader(PUML<Topo>& puml) : m_puml(puml) {
#ifdef USE_MPI
    m_comm = puml.comm();
    MPI_Comm_rank(m_comm, &m_rank);
#endif // USE_MPI
  }

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
    addData<double>(
        "geometry", vertexName, DataType::Vertex, {internal::Topology<Topo>::dimension()});
  }

  void inferSize(DataType type, const std::string& dataset) {
    const auto names = utils::StringUtils::split(dataset, ':');
    if (names.size() != 2) {
      logError() << "Dataset to infer size name must have the form \"filename:/dataset\"";
    }

    // Open the cell file
    hid_t h5plist = H5Pcreate(H5P_FILE_ACCESS);
    checkH5Err(h5plist);
#ifdef USE_MPI
    checkH5Err(H5Pset_fapl_mpio(h5plist, m_comm, MPI_INFO_NULL));
#endif // USE_MPI

    hid_t h5file = H5Fopen(names[0].c_str(), H5F_ACC_RDONLY, h5plist);
    checkH5Err(h5file);

    // Get cell dataset
    hid_t h5dataset = H5Dopen(h5file, names[1].c_str(), H5P_DEFAULT);
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
    checkH5Err(H5Fclose(h5file));

    // Close other H5 stuff
    checkH5Err(H5Pclose(h5plist));
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
    std::vector<std::string> dataNames = utils::StringUtils::split(path, ':');
    if (dataNames.size() != 2) {
      logError() << "Data" << name << "must have the form \"filename:/dataset\", but it has"
                 << path;
    }

    // Open the cell file
    hid_t h5plist = H5Pcreate(H5P_FILE_ACCESS);
    checkH5Err(h5plist);
#ifdef USE_MPI
    checkH5Err(H5Pset_fapl_mpio(h5plist, m_comm, MPI_INFO_NULL));
#endif // USE_MPI

    hid_t h5file = H5Fopen(dataNames[0].c_str(), H5F_ACC_RDONLY, h5plist);
    checkH5Err(h5file);

    const unsigned long totalSize = cellDistributor.totalSize();

    // Get cell dataset
    hid_t h5dataset = H5Dopen(h5file, dataNames[1].c_str(), H5P_DEFAULT);
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

    T* data = m_puml.template allocateData<T>(name,
                                              type,
                                              sizes
#ifdef USE_MPI
                                              ,
                                              mpiType
#endif // USE_MPI
    );
    checkH5Err(H5Dread(h5dataset, hdf5Type, h5memspace, h5space, h5alist, data));

    // Close data
    checkH5Err(H5Sclose(h5space));
    checkH5Err(H5Sclose(h5memspace));
    checkH5Err(H5Dclose(h5dataset));
    checkH5Err(H5Fclose(h5file));

    // Close other H5 stuff
    checkH5Err(H5Pclose(h5plist));
    checkH5Err(H5Pclose(h5alist));
  }

  private:
  template <typename TT>
  static void checkH5ErrImpl(TT status, const char* file, int line, int rank) {
    if (status < 0) {
      logError() << utils::nospace << "An HDF5 error occurred in PUML (" << file << ": " << line
                 << ") on rank " << rank;
    }
  }

  PUML<Topo>& m_puml;
  int m_rank{0};
#ifdef USE_MPI
  MPI_Comm m_comm{MPI_COMM_WORLD};
#endif // USE_MPI
};

#undef checkH5Err

} // namespace PUML

#endif // PUML_HDF5READER_H
