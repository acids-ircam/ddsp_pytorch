
  function(query_literal_where input)
    query_literal("${input}" __query_literal_select_predicate)
    ans(success)

    if(NOT success)
      return()
    endif()

    __query_literal_select_predicate(${ARGN})
    ans(match)
    if(match)
      set(result ${ARGN})
      return_ref(result)
    endif()
    return()
  endfunction()