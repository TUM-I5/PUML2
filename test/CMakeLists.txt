# SPDX-FileCopyrightText: 2026 Technical University of Munich
#
# SPDX-License-Identifier: BSD-3-Clause

include(FetchContent)

set(PUML_GOOGLETEST_TAG "v1.15.2" CACHE STRING "GoogleTest revision to build against")

FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG ${PUML_GOOGLETEST_TAG}
  GIT_SHALLOW TRUE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(puml-tests
  PumlTestMain.cpp
  TestDistributor.cpp
  TestTopology.cpp
  TestMesh.cpp
  TestPartition.cpp
  TestPartitioner.cpp
  TestReader.cpp
  TestClone.cpp
  TestError.cpp
  TestRagged.cpp
  TestMixedReader.cpp
  TestHandle.cpp
  TestLocalize.cpp)

target_link_libraries(puml-tests PRIVATE PUML::Hdf5 GTest::gtest)
target_compile_definitions(puml-tests
  PRIVATE PUML_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

# -Wconversion catches the narrowing between the id and size types. Its
# sign-conversion half is left off: MPI counts ranks in int while the
# containers index in size_t, so it fires on every rank loop without pointing
# at a defect.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(puml-tests PRIVATE -Wall -Wextra -Wconversion -Werror)
endif()

# Built, but not run: they take their input on the command line.
if(PUML_USE_MPI)
  add_executable(puml-example-simple simple-test/puml.cpp)
  target_link_libraries(puml-example-simple PRIVATE PUML::Hdf5)

  add_executable(puml-partition-benchmark PartitionBenchmark.cpp)
  target_link_libraries(puml-partition-benchmark PRIVATE PUML::Hdf5)
  target_include_directories(puml-partition-benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(PUML_USE_MPI)
  foreach(ranks IN ITEMS 1 2 3 4)
    add_test(NAME puml-tests-${ranks}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:puml-tests> ${MPIEXEC_POSTFLAGS})
  endforeach()
else()
  add_test(NAME puml-tests COMMAND puml-tests)
endif()
