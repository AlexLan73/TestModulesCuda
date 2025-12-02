cmake_minimum_required(VERSION 3.18)
project(TestModulesCuda)

enable_language(CUDA)
find_package(CUDA REQUIRED)

# Настройка CUDA architectures в зависимости от ОС
if(NOT CMAKE_CUDA_ARCHITECTURES)
    if(WIN32)
        set(CMAKE_CUDA_ARCHITECTURES "52" CACHE STRING "CUDA architectures: 52 for RTX 2080 Ti on Windows")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "75;86;89" CACHE STRING "CUDA architectures: 75;86;89 for RTX 3060 on Linux")
    endif()
endif()

add_executable(main main.cpp src/RandomGeneratorGPU.cu)
target_include_directories(main PRIVATE src)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})