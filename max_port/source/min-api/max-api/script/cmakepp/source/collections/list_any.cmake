## `[](<list&> <predicate:<[](<any>)->bool>)-><bool>`
##
## returns true if there exists an element in `<list>` for which the `<predicate>` holds
function(list_any __list_any_lst __list_any_predicate)
  function_import("${__list_any_predicate}" as __list_any_predicate REDEFINE)

  foreach(__list_any_item ${${__list_any_lst}})
    __list_any_predicate("${__list_any_item}")
    ans(__list_any_predicate_holds)
    if(__list_any_predicate_holds)
      return(true)
    endif()
  endforeach()
  return(false)
endfunction()


