// SPDX-FileCopyrightText: 2019-2023 Technical University of Munich
//
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @file
 *  This file is part of PUML
 *
 *  For conditions of distribution and use, please see the copyright
 *  notice in the file 'COPYING' at the root directory of this package
 *  and the copyright notice at https://github.com/TUM-I5/PUMGen
 *
 * @author David Schneller <david.schneller@tum.de>
 */

#ifndef PUML_TYPE_INFERENCE_H
#define PUML_TYPE_INFERENCE_H

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include <hdf5.h>

namespace PUML
{

#ifdef USE_MPI
template<typename T> class MPITypeInfer { public: static MPI_Datatype type() { return MPI_BYTE; } };
template<> class MPITypeInfer<char> { public: static MPI_Datatype type() { return MPI_CHAR; } };
template<> class MPITypeInfer<signed char> { public: static MPI_Datatype type() { return MPI_SIGNED_CHAR; } };
template<> class MPITypeInfer<unsigned char> { public: static MPI_Datatype type() { return MPI_UNSIGNED_CHAR; } };
template<> class MPITypeInfer<short> { public: static MPI_Datatype type() { return MPI_SHORT; } };
template<> class MPITypeInfer<unsigned short> { public: static MPI_Datatype type() { return MPI_UNSIGNED_SHORT; } };
template<> class MPITypeInfer<int> { public: static MPI_Datatype type() { return MPI_INT; } };
template<> class MPITypeInfer<unsigned> { public: static MPI_Datatype type() { return MPI_UNSIGNED; } };
template<> class MPITypeInfer<long> { public: static MPI_Datatype type() { return MPI_LONG; } };
template<> class MPITypeInfer<unsigned long> { public: static MPI_Datatype type() { return MPI_UNSIGNED_LONG; } };
template<> class MPITypeInfer<long long> { public: static MPI_Datatype type() { return MPI_LONG_LONG; } };
template<> class MPITypeInfer<unsigned long long> { public: static MPI_Datatype type() { return MPI_UNSIGNED_LONG_LONG; } };
template<> class MPITypeInfer<float> { public: static MPI_Datatype type() { return MPI_FLOAT; } };
template<> class MPITypeInfer<double> { public: static MPI_Datatype type() { return MPI_DOUBLE; } };
template<> class MPITypeInfer<long double> { public: static MPI_Datatype type() { return MPI_LONG_DOUBLE; } };
template<> class MPITypeInfer<wchar_t> { public: static MPI_Datatype type() { return MPI_WCHAR; } };
#endif

template<typename T> class HDF5TypeInfer { public: static hid_t type() { return -1; } };
template<> class HDF5TypeInfer<char> { public: static hid_t type() { return H5T_NATIVE_CHAR; } };
template<> class HDF5TypeInfer<signed char> { public: static hid_t type() { return H5T_NATIVE_SCHAR; } };
template<> class HDF5TypeInfer<unsigned char> { public: static hid_t type() { return H5T_NATIVE_UCHAR; } };
template<> class HDF5TypeInfer<short> { public: static hid_t type() { return H5T_NATIVE_SHORT; } };
template<> class HDF5TypeInfer<unsigned short> { public: static hid_t type() { return H5T_NATIVE_USHORT; } };
template<> class HDF5TypeInfer<int> { public: static hid_t type() { return H5T_NATIVE_INT; } };
template<> class HDF5TypeInfer<unsigned> { public: static hid_t type() { return H5T_NATIVE_UINT; } };
template<> class HDF5TypeInfer<long> { public: static hid_t type() { return H5T_NATIVE_LONG; } };
template<> class HDF5TypeInfer<unsigned long> { public: static hid_t type() { return H5T_NATIVE_ULONG; } };
template<> class HDF5TypeInfer<long long> { public: static hid_t type() { return H5T_NATIVE_LLONG; } };
template<> class HDF5TypeInfer<unsigned long long> { public: static hid_t type() { return H5T_NATIVE_ULLONG; } };
template<> class HDF5TypeInfer<float> { public: static hid_t type() { return H5T_NATIVE_FLOAT; } };
template<> class HDF5TypeInfer<double> { public: static hid_t type() { return H5T_NATIVE_DOUBLE; } };
template<> class HDF5TypeInfer<long double> { public: static hid_t type() { return H5T_NATIVE_LDOUBLE; } };

} // namespace PUML

#endif
