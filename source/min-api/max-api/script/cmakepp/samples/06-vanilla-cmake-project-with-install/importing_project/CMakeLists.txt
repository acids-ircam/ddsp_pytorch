cmake_minimum_required(VERSION 2.8.12)




#find_library(library "install_lib")
#find_path(include_dir "install_lib")
project(project)


add_library(install_lib STATIC IMPORTED)


include_directories(${include_dir})
add_executable(myexe main.cpp)
target_link_libraries(myexe install_lib)