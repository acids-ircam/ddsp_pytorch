
function(t_target_identifier)
  if("${ARGN}" MATCHES "^[a-zA-Z0-9_:]+$")  
    return(true ${ARGN})
  else()
    return(false)
  endif()
endfunction()