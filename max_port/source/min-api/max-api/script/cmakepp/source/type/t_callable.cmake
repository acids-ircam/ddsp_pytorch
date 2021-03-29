
  function(t_callable)
    if(NOT ARGN)
      return(false)
    endif()
    callable("${ARGN}")
    ans(callable)
    return(true ${callable})
  endfunction()
