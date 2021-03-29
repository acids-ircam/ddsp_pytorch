## returns a list of numbers [ start_index, end_index)
## if start_index equals end_index the list is empty
## if end_index is less than start_index then the indices are in declining order
## ie index_range(5 3) => 5 4
## (do not confuse this function with the `range_` functions)
function(index_range start_index end_index)
  
  if(${start_index} EQUAL ${end_index})
    return()
  endif()

  set(result)
  if(${end_index} LESS ${start_index})
    set(increment -1)
    math(EXPR end_index "${end_index} + 1")
  else()
    set(increment 1)
    math(EXPR end_index "${end_index} - 1")
  
  endif()
  
  foreach(i RANGE ${start_index} ${end_index} ${increment})
    list(APPEND result ${i})
  endforeach()
  return(${result})
endfunction()