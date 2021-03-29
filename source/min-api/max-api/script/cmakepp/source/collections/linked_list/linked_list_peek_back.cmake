function(linked_list_peek_back linked_list)
  map_tryget("${linked_list}" tail)
  ans(tail)
  if(NOT tail)
    return()
  endif()

  if("${ARGN}" STREQUAL "--node")
    return(${tail})
  endif() 

  address_get("${tail}")
  return_ans()
endfunction()