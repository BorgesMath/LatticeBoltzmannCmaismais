cmake_minimum_required(VERSION 3.28)
project(LatticeBoltzmann LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler /O2")

add_executable(LatticeBoltzmann
        main.cu
        initialization/initialization.cu
        boundaries/open_boundaries.cu
        post_process/post_process.cu
        lbm/lbm.cu
)

set_target_properties(LatticeBoltzmann PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 86
)