## returns the index of the one of the specified items
## if no element is found then -1 is returned 
## no guarantee is made on which item's index
## is returned 
function(list_find_any __list_find_any_lst )
  foreach(__list_find_any_item ${ARGN})
    list(FIND ${__list_find_any_lst} ${__list_find_any_item} __list_find_any_idx)
    if(${__list_find_any_idx} GREATER -1)
      return(${__list_find_any_idx})
    endif()
  endforeach()
  return(-1)
endfunction()
