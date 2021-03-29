## includes all files identified by globbing expressions
## see `glob` on globbing expressions
function(include_glob)
  set(args ${ARGN})
  glob(${args})
  ans(files)
  foreach(file ${files})
    include_once("${file}")
  endforeach()

  return()
endfunction()