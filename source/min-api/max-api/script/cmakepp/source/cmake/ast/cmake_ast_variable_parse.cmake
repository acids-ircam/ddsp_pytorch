

function(cmake_ast_variable_parse target invocation_token)
  map_tryget(${invocation_token} value)
  ans(value)
  if(NOT "${value}" STREQUAL "set")
    return()
  endif()
  ## get first name token inside invocation_token
  cmake_token_range_find_next_by_type(${invocation_token} "(unquoted_argument)|(quoated_argument)")
  ans(name_token)

  ## get the value of the name token
  map_tryget(${name_token} literal_value)
  ans(name)

  ## get the beginning of values 
  ## the first token after name 
  map_tryget(${name_token} next)
  ans(values_begin)
  ## get the ending of values
  ## the ) 
  cmake_token_range_find_next_by_type(${invocation_token} "(nesting)")
  ans(nesting)
  map_tryget(${nesting} end)
  ans(values_end)
  
  ## set the values
  map_set(${invocation_token} invocation_type variable)
  map_set(${invocation_token} name_token ${name_token})
  map_set(${invocation_token} name ${name})
  map_set(${invocation_token} values_begin ${values_begin})
  map_set(${invocation_token} values_end ${values_end})


  ## add the the variables to the target map
  map_append(${target} variables ${invocation_token})
  map_set(${target} "variable-${name}" ${invocation_token})
endfunction()