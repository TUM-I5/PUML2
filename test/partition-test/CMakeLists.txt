cmake_minimum_required(VERSION 3.18)
project(tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../cmake" ${CMAKE_MODULE_PATH})

find_package(MPI REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(ParMETIS REQUIRED)
find_package(ParHIP REQUIRED)
find_package(PTSCOTCH REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)

add_executable(tests ./puml.cpp)
target_include_directories(tests PRIVATE
        ../..
        ../../submodules
        ${MPI_CXX_INCLUDE_DIRS}
        ${HDF5_INCLUDE_DIRS}
        ${PARMETIS_INCLUDE_DIRS}
        ${PARHIP_INCLUDE_DIRS}
        ${PTSCOTCH_INCLUDE_DIRS})

target_link_libraries(tests
        ${MPI_CXX_LIBRARIES}
        ${HDF5_C_HL_LIBRARIES}
        ${HDF5_C_LIBRARIES}
        ${PARMETIS_LIBRARIES}
        ${PARHIP_LIBRARIES}
        ${PTSCOTCH_LIBRARIES}
        nlohmann_json::nlohmann_json)

target_compile_definitions(tests PRIVATE USE_MPI REAL_SIZE=8)

enable_testing()
add_test(NAME tests COMMAND tests)