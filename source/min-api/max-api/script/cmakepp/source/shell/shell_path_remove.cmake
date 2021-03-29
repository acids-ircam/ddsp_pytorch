

function(shell_path_remove path)
  shell_path_get()
  ans(paths)

  path("${path}")
  ans(path)

  list_contains(paths "${path}")
  ans(res)
  if(res)
    list_remove(paths "${path}")
    shell_path_set(${paths})
    return(true)
  else()
    return(false)
  endif()

endfunction()