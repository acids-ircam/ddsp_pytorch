# removes the last element from list and returns it
function(list_pop_back __list_pop_back_lst)

  if("${${__list_pop_back_lst}}_" STREQUAL "_")
    return()
  endif()
  list(LENGTH "${__list_pop_back_lst}" len)
  math(EXPR len "${len} - 1")
  list(GET "${__list_pop_back_lst}" "${len}" res)
  list(REMOVE_AT "${__list_pop_back_lst}" ${len})
  set("${__list_pop_back_lst}" ${${__list_pop_back_lst}} PARENT_SCOPE)
  return_ref(res)
endfunction()



  # removes the last element from list and returns it
  ## faster version
macro(list_pop_back __list_pop_back_lst)
  if("${${__list_pop_back_lst}}_" STREQUAL "_")
    set(__ans)
  else()
    list(LENGTH "${__list_pop_back_lst}" __list_pop_back_length)
    math(EXPR __list_pop_back_length "${__list_pop_back_length} - 1")
    list(GET "${__list_pop_back_lst}" "${__list_pop_back_length}" __ans)
    list(REMOVE_AT "${__list_pop_back_lst}" ${__list_pop_back_length})
  endif()
endmacro()