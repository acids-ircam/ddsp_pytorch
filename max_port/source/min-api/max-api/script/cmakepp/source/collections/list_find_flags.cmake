## returns a map of all found flags specified as ARGN
##  
function(list_find_flags __list_find_flags_lst)
  map_new()
  ans(__list_find_flags_result)
  foreach(__list_find_flags_itm ${ARGN})
    list(FIND "${__list_find_flags_lst}" "${__list_find_flags_itm}" __list_find_flags_item)
    if(NOT "${__list_find_flags_item}" LESS 0)
      map_set(${__list_find_flags_result} "${__list_find_flags_itm}" true)
    endif()
  endforeach()
  return(${__list_find_flags_result})
endfunction()