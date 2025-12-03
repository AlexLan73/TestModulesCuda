cmake_minimum_required(VERSION 3.18)
project(TestModulesCuda)

# ===== БЛОК ДЛЯ LINUX =====
set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures for Linux")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.0/bin/nvcc" CACHE PATH "CUDA compiler")

# ===== БЛОК ДЛЯ WINDOWS (ЗАКОММЕНТИРОВАН) =====
# set(CMAKE_CUDA_ARCHITECTURES 52 CACHE STRING "CUDA architectures for Windows")
# set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe" CACHE PATH "CUDA compiler")

enable_language(CUDA)
find_package(CUDA REQUIRED)

add_executable(main main.cpp src/RandomGeneratorGPU.cu)
target_include_directories(main PRIVATE src)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
