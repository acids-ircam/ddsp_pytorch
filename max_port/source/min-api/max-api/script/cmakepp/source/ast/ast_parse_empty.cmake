
  function(ast_parse_empty )#definition stream create_node
    map_tryget(${definition}  empty)
    ans(is_empty)
    if(NOT is_empty)
      return(false)
    endif()
   # message("parsed empty!")
    if(NOT create_node)
      return(true)
    endif()

    map_new()
    ans(node)
    return(${node})
  endfunction()