cmake_minimum_required(VERSION 2.8.12)
## CMakeLists.txt for a simple project 
set(current_dir "${CMAKE_CURRENT_SOURCE_DIR}")
## get cmakepp
if(NOT EXISTS "${current_dir}/cmakepp.cmake")
  file(DOWNLOAD "https://github.com/AnotherFoxGuy/cmakepp/releases/download/v0.0.3/cmakepp.cmake" "${current_dir}/cmakepp.cmake")
endif()

include("${current_dir}/cmakepp.cmake")
    
if(NOT EXISTS ${current_dir}/dependencies/eigen3)
 message("installing Eigen3 from bitbucket")
 pull_package(eigen/eigen?tag=3.1.0 dependencies/eigen3)
 ans(package_handle)
 if(NOT package_handle)
  message(FATAL_ERROR "could not pull Eigen3")
 endif()
endif()

## from here on everything can be a normal CMakeLists file
project(sample01)

## include the eigen3 directory so that myexe has access to the header files
include_directories("dependencies/eigen3")

add_executable(myexe "main.cpp")