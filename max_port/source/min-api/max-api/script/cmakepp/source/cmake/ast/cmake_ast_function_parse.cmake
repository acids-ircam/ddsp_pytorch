function(cmake_ast_function_parse target end)
  map_tryget(${end} value)
  ans(value)
  
  if(NOT "${value}" MATCHES "^(endfunction)|(endmacro)$")
    return()
  endif()
  map_tryget(${end} invocation_nesting_begin)
  ans(begin)

  map_tryget(${begin} value)
  ans(command_type)


  ## get name token and name
  cmake_token_range_find_next_by_type(${begin} "(unquoted_argument)|(quoated_argument)")
  ans(name_token)
  map_tryget(${name_token} literal_value)
  ans(name)

  ## next after name is the beginning of the signature
  map_tryget(${name_token} next)
  ans(signature_begin)

  ## next->end is the closing parentheses
  map_tryget(${begin} next)
  ans(parens)
  map_tryget(${parens} end)
  ans(signature_end) ## the closing paren

  ## get the beginning and end of the body
  ## body begins directly after signature
  map_tryget(${signature_end} next)
  ans(body_begin)

  ## body ends directly before endfunction/endmacro
  set(body_end ${end})
  

  ## set the extracted vars
  map_set(${begin} command_type "${command_type}")  
  map_set(${begin} invocation_type command_definition)
  map_set(${begin} name_token ${name_token})
  map_set(${begin} name "${name}")
  map_set(${begin} signature_begin ${signature_begin})
  map_set(${begin} signature_end ${signature_end})
  map_set(${begin} body_begin ${body_begin}) 
  map_set(${begin} body_end ${body_end})

  ## 
  map_append(${target} command_definitions ${begin})
  map_set(${target} "command-${name}" ${begin})
  return_ref(begin)

endfunction()
