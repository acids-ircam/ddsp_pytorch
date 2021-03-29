
  ## pads the list so that every index is set then applies the specified value
  function(list_pad_set lst_ref indices pad_value value)
    
    set(list "${${lst_ref}}")
    set(max -1)
    foreach(i ${indices})
      if(${i} GREATER ${max})
        set(max ${i})
      endif()
    endforeach()
    math(EXPR max "${max} + 1")
    list_pad(list "${max}" "${pad_value}")
    foreach(i ${indices})
      list(INSERT list "${i}" "${value}")
      math(EXPR i "${i} + 1")
      list(REMOVE_AT list "${i}")
    endforeach()
    set(${lst_ref} "${list}" PARENT_SCOPE)
    return()
  endfunction()

