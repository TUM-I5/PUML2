cmake_minimum_required(VERSION 3.9)
project(tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../cmake" ${CMAKE_MODULE_PATH})

find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)

add_executable(tests ./puml.cpp)
target_include_directories(tests PRIVATE
        ../..
        ../../submodules
        ${HDF5_INCLUDE_DIRS}
        ${MPI_CXX_INCLUDE_DIRS}
        )

target_link_libraries(tests
        ${HDF5_C_HL_LIBRARIES}
        ${HDF5_C_LIBRARIES}
        ${MPI_CXX_LIBRARIES})

target_compile_definitions(tests PRIVATE USE_MPI REAL_SIZE=8)

enable_testing()
add_test(NAME tests COMMAND tests)