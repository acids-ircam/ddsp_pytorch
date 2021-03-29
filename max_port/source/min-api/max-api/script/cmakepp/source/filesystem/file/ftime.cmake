
## returns the timestamp for the specified path
function(ftime path)
  path_qualify(path)

  if(NOT EXISTS "${path}")
    return()
  endif()

  file(TIMESTAMP "${path}" res)

  return_ref(res)
endfunction()