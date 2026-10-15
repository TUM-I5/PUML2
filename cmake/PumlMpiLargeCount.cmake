# SPDX-FileCopyrightText: 2026 Technical University of Munich
#
# SPDX-License-Identifier: BSD-3-Clause

# Whether an MPI library has the calls of MPI 4 that count in MPI_Count, which
# PUML uses where it has them (see Exchange.h). Exchange.h tells them from
# MPI_VERSION by itself, but some libraries that have them do not report MPI 4
# (Intel MPI 2021 reports 3.1); for those, the project that builds PUML
# defines PUML_MPI_LARGE_COUNT where this finds the calls -- PUML's own
# CMakeLists.txt for its target, and a project that takes the headers of PUML
# without it on its own.
#
#   puml_mpi_has_large_count(<result> <MPI target>)
#
# sets <result> (cached, as any check result) to whether a program that makes
# these calls compiles and links against <MPI target>, such as MPI::MPI_CXX or
# MPI::MPI_C.

include_guard(GLOBAL)

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

function(puml_mpi_has_large_count result target)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_LIBRARIES ${target})
  check_cxx_source_compiles([[
#include <mpi.h>

int main() {
  MPI_Count count = 0;
  MPI_Aint offset = 0;
  MPI_Request request = MPI_REQUEST_NULL;
  MPI_Alltoallv_c(
      nullptr, &count, &offset, MPI_BYTE, nullptr, &count, &offset, MPI_BYTE, MPI_COMM_WORLD);
  MPI_Isend_c(nullptr, count, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &request);
  MPI_Irecv_c(nullptr, count, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &request);
  return 0;
}
]] ${result})
  cmake_pop_check_state()
  set(${result} "${${result}}" PARENT_SCOPE)
endfunction()
