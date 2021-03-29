## `(<length:<int>> <range...>)-><instanciated range...>`
##
## tries to simplify the specified range for the given length
## his is done by getting the indices and then getting the range from indices
function(range_simplify length)
  set(args ${ARGN})

  list_pop_front(args)
  ans(current_range)

  range_indices("${length}" "${current_range}")
  ans(indices)

  ## get all indices
  while(true)
    list(LENGTH args indices_length)
    if(${indices_length} EQUAL 0)
      break()
    endif()
    list_pop_front(args)
    ans(current_range)
    list_range_get(indices "${current_range}")
    ans(indices)
  endwhile()

  range_from_indices(${indices})
  return_ans()
endfunction()