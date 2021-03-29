function(t_map)
  obj("${ARGN}")
  ans(map)
  if(NOT map)
    return(false)
  endif()
  return(true ${map})
endfunction()



