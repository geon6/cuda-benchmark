cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)

project(cuda-benchmark LANGUAGES CXX CUDA)

include_directories(include)

add_subdirectory(gemm)