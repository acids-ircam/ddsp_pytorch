
function(fexists)
  path("${ARGN}")
  ans(path)

  if(NOT EXISTS "${path}")
    return(false)
  endif()

  if(IS_DIRECTORY "${path}")
    return(false)
  endif()
  return(true)
endfunction()
