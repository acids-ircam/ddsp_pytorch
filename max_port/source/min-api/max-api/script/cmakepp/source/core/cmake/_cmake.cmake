
## fast wrapper for cmake
function(cmake_lean)
  wrap_executable_bare(cmake_lean "${CMAKE_COMMAND}")
  cmake_lean(${ARGN})
  return_ans()
endfunction()