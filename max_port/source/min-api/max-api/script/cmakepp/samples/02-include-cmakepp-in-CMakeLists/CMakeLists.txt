cmake_minimum_required(VERSION 2.8.12)

## be sure to have the compiled cmakepp in the project dir
include("cmakepp.cmake")

## now that cmakepp is available it is ready to use.
project(sample02)

# write a simple main file
fwrite("main.cpp" 
"
#include <iostream>
int main(){
  std::cout << \"hello\" << std::endl;
}")

# and create an executable form it
add_executable(myexe main.cpp)