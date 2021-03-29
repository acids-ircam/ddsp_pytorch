
function(bash_lean)
  wrap_executable_bare(bash_lean bash)
  bash_lean(${ARGN})
  return_ans()
endfunction()