
  function(query_literal_eq input)
    if("${ARGN}" EQUAL "${input}")
      return(true)
    endif()
    return(false)
  endfunction()
