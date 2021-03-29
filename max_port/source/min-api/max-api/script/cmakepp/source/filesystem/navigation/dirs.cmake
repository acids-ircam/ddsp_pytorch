# returns all directories currently on directory stack
# also see pushd popd
function(dirs)
  stack_enumerate(__global_push_d_stack)
  ans(res)
  return_ref(res)
endfunction()