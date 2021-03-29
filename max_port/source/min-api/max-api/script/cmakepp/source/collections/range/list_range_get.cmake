
## returns the elements of the specified list ref which are indexed by specified range
function(list_range_get __lst_ref)
  list(LENGTH ${__lst_ref} __len)
  range_indices("${__len}" ${ARGN})
  ans(__indices)
  list(LENGTH __indices __len)
  if(NOT __len)
    return()
  endif()
  list(GET ${__lst_ref} ${__indices} __res)
  return_ref(__res)
endfunction()
