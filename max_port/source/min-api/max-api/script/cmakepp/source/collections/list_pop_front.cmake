# removes the first value of the list and returns it
function(list_pop_front  __list_pop_front_lst)
  set(res)

  list(LENGTH "${__list_pop_front_lst}" len)
  if("${len}" EQUAL 0)
    return()
  endif()

  list(GET ${__list_pop_front_lst} 0 res)

  if(${len} EQUAL 1) 
    set(${__list_pop_front_lst} )
  else()
    list(REMOVE_AT "${__list_pop_front_lst}" 0)
  endif()
  #message("${__list_pop_front_lst} is ${${__list_pop_front_lst}}")
#  set(${result} ${res} PARENT_SCOPE)
  set(${__list_pop_front_lst} ${${__list_pop_front_lst}} PARENT_SCOPE)
  return_ref(res)
endfunction()


# removes the first value of the list and returns it
## faster version
macro(list_pop_front  __list_pop_front_lst)
  list(LENGTH "${__list_pop_front_lst}" __list_pop_front_length)
  if(NOT "${__list_pop_front_length}" EQUAL 0)
    list(GET ${__list_pop_front_lst} 0 __ans)

    if(${__list_pop_front_length} EQUAL 1) 
      set(${__list_pop_front_lst})
    else()
      list(REMOVE_AT "${__list_pop_front_lst}" 0)
    endif()
  else()
    set(__ans)
  endif()

endmacro()