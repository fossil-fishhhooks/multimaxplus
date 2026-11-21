cmake_minimum_required(VERSION 3.24)

# Honor an explicit target (e.g. -DCMAKE_CUDA_ARCHITECTURES=80 for A100),
# including when configuring on a machine without a GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(mumaxplus LANGUAGES CUDA CXX)

set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (WIN32)
    set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS TRUE)
endif()

set(FP_PRECISION "SINGLE" CACHE STRING "Floating point precision (SINGLE or DOUBLE)")
add_definitions(-DFP_PRECISION=${FP_PRECISION})
set(MUMAX_MODULE_NAME "_mumaxpluscpp_single" CACHE STRING "Name of the mumaxpluscpp module")
add_definitions(-DMUMAX_MODULE_NAME=${MUMAX_MODULE_NAME})

add_definitions(-D_USE_MATH_DEFINES) # Needed for cmath constants on Windows

# Discover shared dependencies here so their imported targets are visible to
# every subdirectory (including the Python bindings and command-line executable).
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(cuFFTMp REQUIRED)

# cuFFTMp's cuFFT headers must precede the ordinary CUDA Toolkit headers.
# Apply the ordering to the bindings and every library that consumes physics.
include_directories(BEFORE "${CUFFTMP_INCLUDE_DIR}")

add_subdirectory(src/bindings)
add_subdirectory(src/cmd)
add_subdirectory(src/core)
add_subdirectory(src/cudautil)
add_subdirectory(src/linsolver)
add_subdirectory(src/physics)
