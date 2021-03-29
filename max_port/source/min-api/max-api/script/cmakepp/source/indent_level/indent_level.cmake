
  function(indent_level)
    map_peek_back(global __indentlevelstack)
    ans(lvl)
    if(NOT lvl)
      return(0)
    endif()
    return_ref(lvl)
  endfunction()