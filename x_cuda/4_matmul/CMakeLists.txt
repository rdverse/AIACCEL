# See apt.kitware.com for updated CMake versions on Ubuntu.
cmake_minimum_required(VERSION 3.30.4)

project(NVBenchDemo CUDA)

# Enable faster builds
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # For tools and IDE integration
set(CMAKE_CUDA_ARCHITECTURES native)   # Build only for current GPU architecture
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)  # Enable separate compilation
set(CMAKE_UNITY_BUILD ON)  # Enable unity builds for faster compilation

# Use Ninja generator if available for faster builds
find_program(NINJA_PATH ninja)
if(NINJA_PATH)
    set(CMAKE_GENERATOR "Ninja" CACHE INTERNAL "" FORCE)
endif()

# Use CPM to fetch NVBench's main branch.
# CPM: https://github.com/cpm-cmake/CPM.cmake
# NVBench: https://github.com/NVIDIA/nvbench
include(cmake/CPM.cmake)
CPMAddPackage("gh:NVIDIA/nvbench#main")

# add data_gen.h
include_directories(${CMAKE_SOURCE_DIR}/../utils)
add_library(data_gen ${CMAKE_SOURCE_DIR}/../utils/data_gen.cu)

add_executable(4_matmul_nvbench 4_matmul_nvbench.cu)
target_link_libraries(4_matmul_nvbench nvbench::main data_gen)
