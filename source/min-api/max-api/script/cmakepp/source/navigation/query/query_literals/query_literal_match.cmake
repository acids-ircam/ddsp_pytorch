

  function(query_literal_match expected)
    #message("match ${expected} - ${ARGN}")
    if("${ARGN}" MATCHES "${expected}")
      return(true)
    endif()
    return(false)
  endfunction()
