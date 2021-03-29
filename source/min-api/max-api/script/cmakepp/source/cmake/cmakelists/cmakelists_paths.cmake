## `(<cmakelists> <file>... [--glob] )-> <relative path>...`
##
## qualifies the paths relative to the cmakelists directory 
## if `--glob` is specified then the `<file>...` will be treated
## as glob expressions
function(cmakelists_paths cmakelists)
  set(args ${ARGN})
  list_extract_flag(args --glob)
  ans(glob)

  map_tryget(${cmakelists} path)
  ans(cmakelists_path)
  path_parent_dir("${cmakelists_path}")
  ans(cmakelists_dir)
  set(files)

  if(glob)
    glob(${args})
    ans(args)
  else()
    paths(${args})
    ans(args)
  endif()
  foreach(file ${args})
    path_relative("${cmakelists_dir}" "${file}")
    ans_append(files)
  endforeach()
  return_ref(files)

endfunction()
