## `(<&list> )`
##
## returns the elements of the specified list ref which are indexed by specified range
function(list_range_try_get __lst_ref)
  list(LENGTH ${__lst_ref} __len)
  # range_indices("${__len}" ${ARGN})
  # ans(__indices2)

  # set(__indices)
  # foreach(__idx ${__indices2})
  #   if(NOT ${__idx} LESS 0 AND ${__idx} LESS ${__len} )
  #     list(APPEND __indices ${__idx})
  #   endif()
  # endforeach()

  range_indices_valid("${__len}" ${ARGN})
  ans(__indices)

  list(LENGTH __indices __len)
  if(NOT __len)
    return()
  endif()
  list(GET ${__lst_ref} ${__indices} __res)
  return_ref(__res)
endfunction()

