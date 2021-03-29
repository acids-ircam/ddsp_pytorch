
## convenience function for accessing cmake
function(cmake)
  wrap_executable(cmake "${CMAKE_COMMAND}")
  cmake(${ARGN})
  return_ans()
endfunction() 
