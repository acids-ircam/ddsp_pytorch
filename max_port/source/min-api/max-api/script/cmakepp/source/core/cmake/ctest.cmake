
## convenience function for accessing ctest
function(ctest)
  path_parent_dir("${CMAKE_COMMAND}")
  ans(parentPath)
  glob("${parentPath}/ctest*")
  ans(ctest_command)

  wrap_executable(ctest "${ctest_command}")
  ctest(${ARGN})
  return_ans()
endfunction() 
