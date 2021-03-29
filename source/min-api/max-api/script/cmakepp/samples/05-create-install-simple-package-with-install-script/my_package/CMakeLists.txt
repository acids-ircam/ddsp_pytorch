cmake_minimum_required(VERSION 2.8.12)


project(my_package)



include_directories(include)

add_library(my_library "src/my_source.cpp" "include/my_header.h")


install(TARGETS my_library EXPORT my_package DESTINATION lib)

install(EXPORT my_package NAMESPACE mp_ DESTINATION lib/my_package)




