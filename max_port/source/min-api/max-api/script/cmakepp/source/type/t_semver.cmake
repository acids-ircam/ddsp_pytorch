function(t_semver)
  semver("${ARGN}")
  ans(res)
  if(NOT res)
    return(false)
  endif()

  return(true ${res})
endfunction()