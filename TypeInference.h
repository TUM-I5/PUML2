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

namespace PUML {

#ifdef USE_MPI
template <typename T>
class MPITypeInfer {
  public:
  static auto type() -> MPI_Datatype { return MPI_BYTE; }
};
template <>
class MPITypeInfer<char> {
  public:
  static auto type() -> MPI_Datatype { return MPI_CHAR; }
};
template <>
class MPITypeInfer<signed char> {
  public:
  static auto type() -> MPI_Datatype { return MPI_SIGNED_CHAR; }
};
template <>
class MPITypeInfer<unsigned char> {
  public:
  static auto type() -> MPI_Datatype { return MPI_UNSIGNED_CHAR; }
};
template <>
class MPITypeInfer<short> {
  public:
  static auto type() -> MPI_Datatype { return MPI_SHORT; }
};
template <>
class MPITypeInfer<unsigned short> {
  public:
  static auto type() -> MPI_Datatype { return MPI_UNSIGNED_SHORT; }
};
template <>
class MPITypeInfer<int> {
  public:
  static auto type() -> MPI_Datatype { return MPI_INT; }
};
template <>
class MPITypeInfer<unsigned> {
  public:
  static auto type() -> MPI_Datatype { return MPI_UNSIGNED; }
};
template <>
class MPITypeInfer<long> {
  public:
  static auto type() -> MPI_Datatype { return MPI_LONG; }
};
template <>
class MPITypeInfer<unsigned long> {
  public:
  static auto type() -> MPI_Datatype { return MPI_UNSIGNED_LONG; }
};
template <>
class MPITypeInfer<long long> {
  public:
  static auto type() -> MPI_Datatype { return MPI_LONG_LONG; }
};
template <>
class MPITypeInfer<unsigned long long> {
  public:
  static auto type() -> MPI_Datatype { return MPI_UNSIGNED_LONG_LONG; }
};
template <>
class MPITypeInfer<float> {
  public:
  static auto type() -> MPI_Datatype { return MPI_FLOAT; }
};
template <>
class MPITypeInfer<double> {
  public:
  static auto type() -> MPI_Datatype { return MPI_DOUBLE; }
};
template <>
class MPITypeInfer<long double> {
  public:
  static auto type() -> MPI_Datatype { return MPI_LONG_DOUBLE; }
};
template <>
class MPITypeInfer<wchar_t> {
  public:
  static auto type() -> MPI_Datatype { return MPI_WCHAR; }
};
#endif

} // namespace PUML

#endif
