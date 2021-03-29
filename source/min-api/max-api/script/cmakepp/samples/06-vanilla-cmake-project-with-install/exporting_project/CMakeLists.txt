cmake_minimum_required(VERSION 2.8.12)


project(install_project)



include_directories(include)

add_library(install_lib "src/install_lib.cpp" "include/install_lib.h")

target_include_directories(install_lib PUBLIC include)

install(TARGETS install_lib  EXPORT install_project DESTINATION lib)
install(EXPORT install_project NAMESPACE ip_  DESTINATION install_project)



