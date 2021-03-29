# creates a temporary directory 
# you can specify an optional parent directory in which it should be created
# usage: mktemp([where])-> <absoute path>
function(mktemp)
  path_temp(${ARGN})
  ans(path)
  mkdir("${path}")
  return_ref(path)
endfunction()
