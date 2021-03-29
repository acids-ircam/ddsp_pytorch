
function(linux_ps_lean)
  wrap_executable_bare(linux_ps_lean ps)
  linux_ps_lean(${ARGN})
  return_ans()
endfunction()