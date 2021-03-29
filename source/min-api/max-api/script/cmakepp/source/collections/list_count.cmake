## `(<list&> <predicate:<[](<any>)-><bool>>> )-><uint>`
##
## counts all element for which the predicate holds 
function(list_count __list_count_lst __list_count_predicate)
  function_import("${__list_count_predicate}" as __list_count_predicate REDEFINE)
  set(__list_count_counter 0)
  foreach(__list_count_item ${${__list_count_lst}})
    __list_count_predicate("${__list_count_item}")
    ans(__list_count_match)
    if(__list_count_match)
      math(EXPR __list_count_counter "${__list_count_counter} + 1") 
    endif()
  endforeach()
  return("${__list_count_counter}")
endfunction()



