cmake_minimum_required(VERSION 2.8.12)

## be sure to have the compiled cmakepp in the project dir
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmakepp.cmake")
    file(DOWNLOAD "https://github.com/AnotherFoxGuy/cmakepp/releases/download/v0.0.3/cmakepp.cmake" "${CMAKE_SOURCE_DIR}/cmakepp.cmake")
endif()
include("cmakepp.cmake")

## now that cmakepp is available it is ready to use.
## and you can use it in you normal CMakeLists.txt
project(sample03)

# write a simple main file
fwrite("main.cpp" "
#include <iostream>
int main(){
  std::cout << \"hello\" << std::endl;
}")

# and create an executable form it
add_executable(myexe main.cpp)