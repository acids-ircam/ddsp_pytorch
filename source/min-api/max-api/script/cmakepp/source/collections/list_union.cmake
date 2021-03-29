# returns a list containing the unqiue set of all elements
# contained in passed list referencese
function(list_union)
  if(NOT ARGN)
    return()
  endif()
  set(__list_union_result)
  foreach(__list_union_list ${ARGN})
    list(APPEND __list_union_result ${${__list_union_list}})
  endforeach() 

  list(REMOVE_DUPLICATES __list_union_result)
  return_ref(__list_union_result)
endfunction()
