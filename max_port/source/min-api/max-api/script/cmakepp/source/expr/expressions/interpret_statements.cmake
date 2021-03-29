function(interpret_statements tokens)
  interpret_separation("${tokens}" "semicolon" " " "" "")
  ans(separation)

  if(NOT separation)
    return(${separation})
  endif()

  map_peek_back("${separation}" elements)
  ans(last)

  map_tryget("${last}" argument)
  ans(last_argument)


  
  set(ast ${separation})
  map_set(${ast} argument "${last_argument}")


  return(${ast})
endfunction()