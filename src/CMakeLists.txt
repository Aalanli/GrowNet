cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(main LANGUAGES CXX CUDA)

set(CMAKE_PREFIX_PATH "/home/allan/Documents/C++/libtorch/")
set(CMAKE_CUDA_FLAGS "-arch=sm_80 -lineinfo -Xcompiler -rdynamic")
set(PYTHON_INCLUDE_DIRS "/home/allan/.conda/envs/Torch/include/python3.8")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(main "main.cc")
add_subdirectory(utils)

#add_compile_definitions(IS_EXECUTABLE)
set_property(TARGET main PROPERTY CXX_STANDARD 14)
set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)
