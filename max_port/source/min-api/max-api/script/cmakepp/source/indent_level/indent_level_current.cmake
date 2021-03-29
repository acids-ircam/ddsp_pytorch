## returns the current index level index which can be used to 
## restore the index level to a specific point
  function(indent_level_current)
    map_property_length(global __indentlevelstack)
    ans(idx)
    math(EXPR idx "${idx} -1")
    if("${idx}" LESS 0)
      set(idx 0)
    endif()
    return_ref(idx)
  endfunction()