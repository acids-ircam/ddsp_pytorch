## `(<listA&:<any...> <listB&:<any...>>)-><any..>`
## 
## 
function(set_difference __set_difference_listA __set_difference_listB)
  if("${${__set_difference_listA}}_" STREQUAL "_")
    return()
  endif()

  if(NOT "${${__set_difference_listB}}_" STREQUAL "_")
    list(REMOVE_ITEM "${__set_difference_listA}" ${${__set_difference_listB}})
  endif()
  list(REMOVE_DUPLICATES ${__set_difference_listA})
  #foreach(__list_operation_item ${${__set_difference_listB}})
   # list(REMOVE_ITEM ${__set_difference_listA} ${__list_operation_item})
  #endforeach()
  return_ref(${__set_difference_listA})
endfunction()


