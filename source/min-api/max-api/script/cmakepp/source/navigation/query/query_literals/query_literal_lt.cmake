
  function(query_literal_lt input)
    if("${ARGN}" LESS "${input}")
      return(true)
    endif()
    return(false)
  endfunction()