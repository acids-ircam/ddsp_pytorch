# Compiling a simple progam by including `cmakepp` and pulling `eigen` library 

## Description

This sample contains a single `CMakeLists.txt` which downloads and includes `cmakecpp` then it downloads `Eigen3` from bitbucket.

## Prerequisites

* CMake version `>=2.8.12`
* `bash`, `powershell` or `cmd.exe` 


## How to use

```bash
sample folder/> mkdir build 
sample folder/> cd build                # creates a build folder 
sample folder/build/> cmake -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=bin -DCMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG=bin ..          # generates the project configuration specified that the executable will be in the build/bin folder
-- Building for: Visual Studio 12 2013
-- Found Git: C:/Program Files (x86)/Git/cmd/git.exe (found version "1.9.0.msysgit.0") 
-- Found Hg: C:/Program Files/Mercurial/hg.exe (found version "3.0") 
-- Found Subversion: C:/Program Files/SlikSvn/bin/svn.exe (found version "1.8.11") 
-- The C compiler identification is ;MSVC 18.0.31010.0
-- The CXX compiler identification is ;MSVC 18.0.31010.0
-- Check for working C compiler using: Visual Studio 12 2013
-- Check for working C compiler using: Visual Studio 12 2013 -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working CXX compiler using: Visual Studio 12 2013
-- Check for working CXX compiler using: Visual Studio 12 2013 -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: C:/Temp/cutil/temp/mktemp_idzo0/build

sample folder/build/> cmake --build .   # builds the generated project 
... CMake build output ...
sample folder/build/> bin/myexe 
  3  -1
2.5 1.5

```

*main.cpp*: 
```cpp
#include <iostream>
#include <Eigen/Dense>
using Eigen::MatrixXd;
int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
}
```

*CMakeLists.txt*: 
```cmake
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
```

