
  function(query_literal_strequal expected)
    #message("strequal ${expected} - ${ARGN}")
    if("${expected}_" STREQUAL "${ARGN}_")
      return(true)
    endif()
    return(false)
  endfunction()