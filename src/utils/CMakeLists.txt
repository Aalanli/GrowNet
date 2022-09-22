cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

target_sources(main
    PUBLIC
        func_constructor.h
        torch_utils.h
        utils.h
        type_repr.h
)

find_package(Torch REQUIRED)
target_link_libraries(main PRIVATE ${TORCH_LIBRARIES})