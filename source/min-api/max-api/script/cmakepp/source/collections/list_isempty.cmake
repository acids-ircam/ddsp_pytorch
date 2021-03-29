# checks if the given list reference is an empty list
  function(list_isempty __list_empty_lst)
    list(LENGTH  ${__list_empty_lst} len)
    if("${len}" EQUAL 0)
      return(true)
    endif()
    return(false)
  endfunction()