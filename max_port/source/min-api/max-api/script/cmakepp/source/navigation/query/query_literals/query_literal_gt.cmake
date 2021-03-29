

  function(query_literal_gt input)
    if("${ARGN}" GREATER "${input}")
      return(true)
    endif()
    return(false)
  endfunction()
