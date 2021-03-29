# removes all items specified in varargs from list
# returns the number of items removed
function(list_remove __list_remove_lst)
  list(LENGTH "${__list_remove_lst}" __lst_len)
  list(LENGTH ARGN __arg_len)
  if(__arg_len EQUAL 0 OR __lst_len EQUAL 0)
    return()
  endif()
  list(REMOVE_ITEM "${__list_remove_lst}" ${ARGN})
  list(LENGTH "${__list_remove_lst}" __lst_new_len)
  math(EXPR __removed_item_count "${__lst_len} - ${__lst_new_len}")
  set("${__list_remove_lst}" "${${__list_remove_lst}}" PARENT_SCOPE)
  return_ref(__removed_item_count)
endfunction()