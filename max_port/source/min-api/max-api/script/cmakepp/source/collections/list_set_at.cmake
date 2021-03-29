# sets the lists value at index to the specified value
# the index is normalized -> negativ indices count down from back of list 
  function(list_set_at __list_set_lst index value)
    if("${index}" EQUAL -1)
      #insert element at end
      list(APPEND ${__list_set_lst} ${value})
      set(${__list_set_lst} ${${__list_set_lst}} PARENT_SCOPE)
      return(true)
    endif()
    list_normalize_index(${__list_set_lst} "${index}")
    ans(index)
    if(index LESS 0)
      return(false)
    endif()
    list_replace_at(${__list_set_lst} "${index}" "${value}")

    set(${__list_set_lst} ${${__list_set_lst}} PARENT_SCOPE)
    return(true)
  endfunction()