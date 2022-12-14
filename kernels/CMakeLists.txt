cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
# project(main LANGUAGES CXX CUDA)
project(main LANGUAGES CXX)

set(CMAKE_PREFIX_PATH "/home/allan/Documents/C++/libtorch/")
#set(CUDNN_INCLUDE_PATH "/usr/lib/cuda/include")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(main "main.cc")
add_subdirectory(utils)

target_link_libraries(main PRIVATE ${TORCH_LIBRARIES})
set_property(TARGET main PROPERTY CXX_STANDARD 20)