# replaces the current working directory with
# the top element of the directory stack_pop and
# removes the top element
function(popd)
  stack_pop(__global_push_d_stack)
  ans(pwd)

  cd("${pwd}" ${ARGN})
  return_ans()
endfunction()
