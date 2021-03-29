
  function(t_uri)
    uri(${ARGN})
    ans(uri)
    if(uri)
      return(true ${uri})

    else()
      return(false)
    endif()
  endfunction()