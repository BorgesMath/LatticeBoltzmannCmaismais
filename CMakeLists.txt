# Define uma versão estável e real do CMake
cmake_minimum_required(VERSION 3.28)

# É estritamente mandatório declarar CXX para que o compilador host (MSVC) seja ativado junto ao nvcc
project(LatticeBoltzmann LANGUAGES CXX CUDA)

# Padronização da linguagem. 
# O C++17 é a norma acadêmica e industrial mais estável para desenvolvimento CUDA atual.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Força a otimização de tempo de execução. Nunca rode LBM em Debug na GPU.
set(CMAKE_BUILD_TYPE Release)

# -O3 otimiza o código do device (GPU)
# -Xcompiler /O2 passa a otimização agressiva para o compilador host no Windows (MSVC)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler /O2")

# Declaração do executável com múltiplos arquivos de código-fonte
add_executable(LatticeBoltzmann
        main.cu
        initialization/initialization.cu
        multiphase/cahn_hilliard.cu
        Magnetismo/poisson.cu
        boundaries/open_boundaries.cu
        post_process/post_process.cu
        lbm/lbm.cu
)

# Propriedades do alvo de compilação
set_target_properties(LatticeBoltzmann PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        # Arquitetura exata da RTX 3050 (Ampere)
        CUDA_ARCHITECTURES 86
)

