cmake_minimum_required(VERSION 3.0)

project(DDSP)

find_package(Torch REQUIRED)

file(GLOB SRC *.cpp)

add_library(ddsp SHARED ${SRC})
target_link_libraries(ddsp "${TORCH_LIBRARIES}")
set_property(TARGET ddsp PROPERTY CXX_STANDARD 11)

set_target_properties(ddsp PROPERTIES PREFIX "" SUFFIX "~.pd_linux")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
