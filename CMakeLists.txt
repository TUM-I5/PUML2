# SPDX-FileCopyrightText: 2026 Technical University of Munich
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.20)

project(PUML
        VERSION 2.0.0
        DESCRIPTION "Parallel Unstructured Mesh Library"
        LANGUAGES CXX)

option(PUML_USE_MPI "Distribute the mesh over MPI ranks" ON)
option(PUML_USE_HDF5 "Build the HDF5 reader, exported as PUML::Hdf5" ON)
option(PUML_TESTING "Build the test suite" OFF)

set(PUML_PARTITIONER "none" CACHE STRING "Graph partitioner to link against")
set_property(CACHE PUML_PARTITIONER PROPERTY STRINGS none parmetis ptscotch parhip)

add_library(PUML INTERFACE)
add_library(PUML::PUML ALIAS PUML)

target_compile_features(PUML INTERFACE cxx_std_17)
target_include_directories(PUML INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/submodules>)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/submodules/utils/logger.h")
  message(FATAL_ERROR
    "submodules/utils is missing. Run: git submodule update --init --recursive")
endif()

if(PUML_USE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_compile_definitions(PUML INTERFACE USE_MPI)
  target_link_libraries(PUML INTERFACE MPI::MPI_CXX)
endif()

# The mesh itself needs no HDF5; only the reader does.
if(PUML_USE_HDF5)
  set(HDF5_PREFER_PARALLEL ${PUML_USE_MPI})
  find_package(HDF5 REQUIRED COMPONENTS C)

  add_library(PUML-hdf5 INTERFACE)
  add_library(PUML::Hdf5 ALIAS PUML-hdf5)
  target_link_libraries(PUML-hdf5 INTERFACE PUML::PUML ${HDF5_C_LIBRARIES})
  target_include_directories(PUML-hdf5 INTERFACE ${HDF5_C_INCLUDE_DIRS})

  if(PUML_USE_MPI AND NOT HDF5_IS_PARALLEL)
    message(WARNING
      "HDF5 was built without MPI support. Reading a mesh will fall back to "
      "the serial driver on every rank.")
  endif()
endif()

if(NOT PUML_PARTITIONER STREQUAL "none")
  if(NOT PUML_USE_MPI)
    message(FATAL_ERROR "PUML_PARTITIONER=${PUML_PARTITIONER} needs PUML_USE_MPI=ON")
  endif()

  if(PUML_PARTITIONER STREQUAL "parmetis")
    set(_puml_partitioner_header parmetis.h)
    set(_puml_partitioner_libs parmetis metis)
    set(_puml_partitioner_define USE_PARMETIS)
  elseif(PUML_PARTITIONER STREQUAL "ptscotch")
    set(_puml_partitioner_header ptscotch.h)
    set(_puml_partitioner_libs ptscotch ptscotcherr scotch)
    set(_puml_partitioner_define USE_PTSCOTCH)
  elseif(PUML_PARTITIONER STREQUAL "parhip")
    set(_puml_partitioner_header parhip_interface.h)
    set(_puml_partitioner_libs parhip_interface)
    set(_puml_partitioner_define USE_PARHIP)
  endif()

  find_path(PUML_PARTITIONER_INCLUDE_DIR NAMES ${_puml_partitioner_header} REQUIRED)
  target_include_directories(PUML INTERFACE ${PUML_PARTITIONER_INCLUDE_DIR})
  target_compile_definitions(PUML INTERFACE ${_puml_partitioner_define})

  foreach(_lib IN LISTS _puml_partitioner_libs)
    find_library(PUML_PARTITIONER_LIB_${_lib} NAMES ${_lib} REQUIRED)
    target_link_libraries(PUML INTERFACE ${PUML_PARTITIONER_LIB_${_lib}})
  endforeach()
endif()

if(PUML_TESTING)
  if(NOT PUML_USE_HDF5)
    message(FATAL_ERROR "PUML_TESTING needs PUML_USE_HDF5=ON for the reader tests")
  endif()
  enable_testing()
  add_subdirectory(test)
endif()
