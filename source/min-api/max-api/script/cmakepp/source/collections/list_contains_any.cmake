
function(list_contains_any __lst)
    if("${ARGC}" EQUAL "1")
    ## no items specified 
    return(true)
  endif()

  list(LENGTH ${__lst} list_len)
  if(NOT list_len)
    ## list is empty and items are specified -> list does not contain
    return(false)
  endif()


  foreach(item ${ARGN})
    list(FIND ${__lst} ${item} idx)
    if(idx GREATER -1)
      return(true)
    endif()

  endforeach() 
  return(false)
endfunction()