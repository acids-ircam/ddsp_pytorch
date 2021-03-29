
# 
function(shell_path_add path)
  set(args ${ARGN})
  list_extract_flag(args "--prepend")
  ans(prepend)

  shell_path_get()
  ans(paths)
  path("${path}")
  ans(path)
  list_contains(paths "${path}")
  ans(res)
  if(res)
    return(false)
  endif()


  if(prepend)
    set(paths "${path};${paths}")
  else()
    set(paths "${paths};${path}")
  endif()

  shell_path_set(${paths})

  return(true)
endfunction()