cmake_minimum_required(VERSION 3.0)
project(ddsp_external)

find_package(Torch REQUIRED)

if (APPLE)
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -undefined dynamic_lookup -stdlib=libc++")
set(EXTERNAL_SUFFIX ".pd_darwin")
set(EXTERNAL_DESTINATION "~/Documents/Pd/externals")
endif()

if (UNIX AND NOT APPLE)
set(EXTERNAL_SUFFIX ".pd_linux")
set(EXTERNAL_DESTINATION "~/Documents/Pd/externals")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_library(ddsp_tilde SHARED ddsp_tilde/ddsp_tilde.cpp ddsp_tilde/ddsp_model.cpp)
target_link_libraries(ddsp_tilde PUBLIC "${TORCH_LIBRARIES}")
set_target_properties(ddsp_tilde PROPERTIES PREFIX "" OUTPUT_NAME "ddsp~" SUFFIX "${EXTERNAL_SUFFIX}")
set_property(TARGET ddsp_tilde PROPERTY CXX_STANDARD 14)

install(TARGETS ddsp_tilde DESTINATION "${EXTERNAL_DESTINATION}")
