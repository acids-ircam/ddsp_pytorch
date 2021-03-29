# applies the glob expressions (passed as varargs)
# to the first n parent directories starting with the current dir
# order of result is in deepest path first
# 0 searches parent paths up to root
# warning do not use --recurse and unlimited depth as it would probably take forever
# @todo extend to quit search when first result is found
function(glob_up n)
  set(args ${ARGN})

  # extract dir
  set(path)
  path("${path}")
  ans(path)

  set(globs ${args})

  # /tld is appended because only its parent dirs are gotten 
  path_parent_dirs("${path}/tld" ${n})
  ans(parent_dirs)

  set(all_matches)
  foreach(parent_dir ${parent_dirs})
    glob("${parent_dir}" ${globs})
    ans(matches)
    list(APPEND all_matches ${matches})
  endforeach()
  return_ref(all_matches)
endfunction()
