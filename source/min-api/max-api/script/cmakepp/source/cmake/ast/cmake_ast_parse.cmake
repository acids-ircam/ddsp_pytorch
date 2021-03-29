## `(<code:<cmake code>|<cmake token...>>)-><cmake ast>`
##
## generates the an ast for the cmake code 
function(cmake_ast_parse code)
  cmake_ast_nesting_pairs()
  ans(nesting_start_end_pairs)
  
  map_values(${nesting_start_end_pairs})
  ans(endings)
  map_keys(${nesting_start_end_pairs})
  ans(openings)
  list_remove_duplicates(endings)

  cmake_tokens("${code}")
  ans(tokens)

  ans_extract(current_invocation)

  map_new()
  ans(ast)

  map_set(${ast} tokens ${tokens})

  ## push the first nesting on the nestings stack
  set(nestings "${ast}")
  set(current_nesting ${ast})

  while(true)
    cmake_token_range_find_next_by_type("${current_invocation}" command_invocation)
    ans(current_invocation)
    if(NOT current_invocation)
      break()
    endif()
    
    
    map_append(${current_nesting} command_invocations ${current_invocation})
    map_tryget(${current_invocation} value)
    ans(invocation_value)

    list_contains(openings ${invocation_value})
    ans(is_opening)
    list_contains(endings ${invocation_value})
    ans(is_closing)

    ## handles the closing of an invocation nesting
    if(is_closing)
      set(begin "${current_nesting}")
      set(end "${current_invocation}")

      ## pop the top nesting
      list(REMOVE_AT nestings 0)
      list(GET nestings 0 current_nesting)

      if("${begin}" STREQUAL "${root_nesting}")  
        message(FORMAT "unbalanced code nesting for {current_invocation.value} @{current_invocation.line}:{current_invocation.column}")
        error( "unbalanced code nesting for {current_invocation.value} @{current_invocation.line}:{current_invocation.column}")
        return()
      endif()


      map_tryget(${begin} value)
      ans(begin_value)
      set(end_value ${invocation_value})

      map_tryget(${nesting_start_end_pairs} ${begin_value})
      ans(current_closings)
      
      list_contains(current_closings ${end_value})
      ans(correct_closing)
      if(NOT correct_closing)
        message(FORMAT "invalid closing for opening '{current_nesting.value}' @{current_nesting.line}:{current_nesting.column}: '{current_invocation.value}' @{current_invocation.line}:{current_invocation.column}")
        error("invalid closing for {current_invocation.value} @{current_invocation.line}:{current_invocation.column}")
        return()
      endif()
            
      map_set(${begin} invocation_nesting_end ${end})
      map_set(${end} invocation_nesting_begin ${begin})
    endif()
    list(LENGTH nestings nesting_depth)
    math(EXPR nesting_depth "${nesting_depth} - 1")
    map_set(${current_invocation} invocation_nesting_depth ${nesting_depth})
    cmake_ast_function_parse("${current_nesting}" "${current_invocation}")
    cmake_ast_variable_parse("${current_nesting}" "${current_invocation}")
    map_append(${current_nesting} children ${current_invocation})

    ## handles the opening of an invocation nesting
    if(is_opening)
      ## push the the current_invocation nesting
      list(INSERT nestings 0 ${current_invocation})
      set(current_nesting ${current_invocation})
    endif()

    map_tryget(${current_invocation} next)
    ans(current_invocation)
  endwhile()

  return_ref(ast)
endfunction()