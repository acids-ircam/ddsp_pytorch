function(linked_list_peek_front linked_list)
  map_tryget("${linked_list}" head)
  ans(head)
  if(NOT head)
    return()
  endif()

  if("${ARGN}" STREQUAL "--node")
    return(${head})
  endif()    

  address_get("${head}")
  return_ans()
endfunction()
