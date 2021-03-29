# wraps the bash executable in cmake
function(bash)
  wrap_executable(bash bash)
  bash(${ARGN})
  return_ans()
endfunction()

