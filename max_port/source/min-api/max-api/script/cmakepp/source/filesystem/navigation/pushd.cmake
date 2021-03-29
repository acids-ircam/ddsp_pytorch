# pushes the specfied directory (or .) onto the 
# directory stack
function(pushd)
  pwd()
  ans(pwd)
  stack_push(__global_push_d_stack "${pwd}")
  if(ARGN)
    cd(${ARGN})
    return_ans()
  endif()
  return_ref(pwd)
endfunction()