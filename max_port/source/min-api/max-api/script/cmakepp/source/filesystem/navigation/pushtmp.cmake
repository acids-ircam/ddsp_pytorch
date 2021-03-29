## `(<?parent dir>)-><qualified path>, path_stack is pushed`
##
## pushes a temporary directory on top of the pathstack 
function(pushtmp)
  mktemp(${ARGN})
  ans(dir)
  pushd("${dir}")
  return_ans()
endfunction()
