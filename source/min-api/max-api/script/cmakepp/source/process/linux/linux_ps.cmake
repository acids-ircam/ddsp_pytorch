# wraps the linux ps command into an executable 
function(linux_ps)
  wrap_executable(linux_ps ps)
  linux_ps(${ARGN})
  return_ans()
endfunction()