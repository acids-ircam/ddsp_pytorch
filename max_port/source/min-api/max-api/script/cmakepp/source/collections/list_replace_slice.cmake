
  ## replaces the specified slice with the specified varargs
  ## returns the elements which were removed
  function(list_replace_slice __list_ref __start_index __end_index)
    ## normalize indices
    list_normalize_index(${__list_ref} ${__start_index})
    ans(__start_index)
    list_normalize_index(${__list_ref} ${__end_index})
    ans(__end_index)


    list(LENGTH ARGN __insert_count)
    ## add new elements
    if(__insert_count)
      list(LENGTH ${__list_ref} __old_length)
      if("${__old_length}" EQUAL "${__start_index}")
        list(APPEND ${__list_ref} ${ARGN})
      else()
        list(INSERT ${__list_ref} ${__start_index} ${ARGN})
      endif()
      math(EXPR __start_index "${__start_index} + ${__insert_count}")
      math(EXPR __end_index "${__end_index} + ${__insert_count}")
    endif()
    
    ## generate index list of elements to remove
    index_range(${__start_index} ${__end_index})
    ans(__indices)

    ## get number of elements to remove
    list(LENGTH __indices __remove_count)
    
    ## get slice which is to be removed and remove it
    set(__removed_elements)
    if(__remove_count)
      list(GET ${__list_ref} ${__indices} __removed_elements)
      list(REMOVE_AT ${__list_ref} ${__indices})
    endif()
    

    ## set result
    set(${__list_ref} ${${__list_ref}} PARENT_SCOPE)
    return_ref(__removed_elements)
  endfunction()