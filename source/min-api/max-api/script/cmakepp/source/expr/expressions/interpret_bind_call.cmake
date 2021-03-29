

  ##
  ## 
  ## <lhs rvalue>::<rhs rvalue>(<parameter>...)  
  ##  => calls the function pointed to by rvalue  using lhs_ rvalue as its first argument
  ## then appending the other parameters
  ## e.g  `'123'::string_length()->3`
  function(interpret_bind_call tokens)
    set(lhs_tokens ${tokens})

    list_pop_back(lhs_tokens)
    ans(paren_token)
    list_pop_back(lhs_tokens)
    ans(callable_token)
    list_pop_back(lhs_tokens)
    ans(colon1)
    list_pop_back(lhs_tokens)
    ans(colon2)


    if(NOT paren_token)
      throw("invalid token count, expected at least 5 got 0" --function interpret_bind_call)
    endif()
    map_tryget("${paren_token}" type)
    ans(paren_type)


    if(NOT "${paren_type}" STREQUAL "paren")
      throw("expected paren token, got ${paren_type}" --function interpret_bind_call)
    endif()


    if(NOT callable_token)
      throw("invalid token count, expected at least 5 got 1" --function interpret_bind_call)
    endif()

    if(NOT colon1)
      throw("invalid token count, expected at least 5 got 2" --function interpret_bind_call)
    endif()
    if(NOT colon2)
      throw("invalid token count, expected at least 5 got 3" --function interpret_bind_call)
    endif()

    if(NOT lhs_tokens)
      throw("invalid token count, expected at least 5 got 4" --function interpret_bind_call)
    endif()

    map_tryget("${colon1}" type)
    ans(colon1_type)
    if(NOT "${colon1_type}" STREQUAL "colon" )
      throw("expected colon" --function interpret_bind_call)
    endif()

    map_tryget("${colon2}" type)
    ans(colon2_type)
    if(NOT "${colon2_type}" STREQUAL "colon" )
      throw("expected second colon" --function interpret_bind_call)
    endif()


    interpret_rvalue("${lhs_tokens}")
    rethrow()
    ans(lhs)


    interpret_rvalue("${callable_token}")
    rethrow()
    ans(callable)

    map_tryget("${paren_token}" tokens)
    ans(parameter_tokens)

    interpret_elements("${parameter_tokens}" "comma" "interpret_ellipsis;interpret_reference_parameter;interpret_expression")
    ans(parameters)

    next_id()
    ans(ref)

    interpret_call_create_code("${ref}" "${callable}" "${lhs};${parameters}")
    ans(code)

    ast_new(
      "${tokens}"
      "bind_call"
      "any"
      "${ref}"
      "${code}"
      "\${${ref}}"   #value
      "false"         # const
      "false"          #pure value
      "${lhs};${callable};${parameters}" #"${lhs};${callable};${parameters}"
      )
    ans(ast)
    return_ref(ast)    

  endfunction()