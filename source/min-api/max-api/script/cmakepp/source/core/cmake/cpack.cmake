
## convenience function for accessing cpack
function(cpack)
  path_parent_dir("${CMAKE_COMMAND}")
  ans(parentPath)
  glob("${parentPath}/cpack*")
  ans(cpack_command)

  wrap_executable(cpack "${cpack_command}")
  cpack(${ARGN})
  return_ans()
endfunction() 
