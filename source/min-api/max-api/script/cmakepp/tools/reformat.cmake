# compiles cmake to the specified target directory
# call this script with cmake -P compile.cmake path/to/target/dir
include("${CMAKE_CURRENT_LIST_DIR}/../cmakepp.cmake")

wrap_executable_bare(cmake_format "cmake-format")

assign(cmake_files = glob("${CMAKE_CURRENT_LIST_DIR}/../source/**.cmake" --relative --recurse))#

foreach(cfile ${cmake_files})
    message(STATUS "formating ${cfile}")
    cmake_format("-i" "${cfile}")
  endforeach()

message("done")