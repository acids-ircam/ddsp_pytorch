  ## sets every element included in range to specified value
  ## 
  function(list_range_set __lst __range __value)
    list_range_indices(${__lst} "${__range}")
    ans(indices)
    foreach(i ${indices})
      list(INSERT "${__lst}" "${i}" "${__value}")
      math(EXPR i "${i} + 1")
      list(REMOVE_AT "${__lst}" "${i}")
    endforeach()
    set(${__lst} ${${__lst}} PARENT_SCOPE)
    return()
  endfunction()
