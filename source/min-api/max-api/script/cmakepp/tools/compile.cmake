# compiles cmake to the specified target directory
# call this script with cmake -P compile.cmake path/to/target/dir
include("${CMAKE_CURRENT_LIST_DIR}/../cmakepp.cmake")

cmakepp_compile("${CMAKE_CURRENT_SOURCE_DIR}/release/cmakepp.cmake")

message("Building release/cmakepp.cmake done")