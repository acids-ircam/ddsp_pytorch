# gets the first element of the list without modififying it
function(list_peek_front __list_peek_front_lst)
  if("${${__list_peek_front_lst}}_" STREQUAL "_")
    return()
  endif()
  list(GET "${__list_peek_front_lst}" 0 res)
  return_ref(res)
endfunction()