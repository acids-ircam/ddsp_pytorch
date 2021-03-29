## `(<list&> <predicate:<[](<any>)->bool>>)-><bool>` 
##
## returns true iff predicate holds for all elements of `<list>` 
## 
function(list_all __list_all_lst __list_all_predicate)
  function_import("${__list_all_predicate}" as __list_all_predicate REDEFINE)
  foreach(it ${${__list_all_lst}})
    __list_all_predicate("${it}")
    ans(__list_all_match)
    if(NOT __list_all_match)
      return(false)
    endif()
  endforeach()
  return(true)
endfunction()