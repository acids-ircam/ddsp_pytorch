function(tar_lean)
  cmake_lean(-E tar ${ARGN})
  return_ans()
endfunction()