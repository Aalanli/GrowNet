cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
# project(main LANGUAGES CXX CUDA)
project(main LANGUAGES CXX)

set(CMAKE_PREFIX_PATH "/home/allan/Documents/C++/libtorch/")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(main lib.h "main.cc")
add_subdirectory(utils)

set_property(TARGET main PROPERTY CXX_STANDARD 20)
