function(interpret_indexation tokens)
  set(lhs_tokens ${tokens})
  list(LENGTH lhs_tokens token_count)
  if(NOT "${token_count}" GREATER 1)
    throw("invalid token count, expected more than one token, got ${token_count}" --function interpret_indexation)
  endif()

  list_pop_back(lhs_tokens)
  ans(indexation_token)

  map_tryget("${indexation_token}" type)
  ans(token_type)


  if(NOT "${token_type}" STREQUAL "bracket")
    throw("invalid token type, expected bracket but got ${token_type}" --function interpret_indexation)
  endif()

  map_tryget("${indexation_token}" tokens)
  ans(inner_tokens)


  interpret_elements("${inner_tokens}" "comma" "interpret_range;interpret_rvalue")
  rethrow()
  ans(elements)

  #print_vars(lhs_tokens)
  interpret_rvalue("${lhs_tokens}")
  rethrow()
  ans(lhs)



  ast_new(
    "${tokens}"         # tokens
    "indexation"        # expression_type
    "${value_type}"                  # value_type
    "${ref}"                  # ref
    "${code}"                  # code
    "\${${ref}}"                  # value
    "false"                  # const
    "false"                  # pure_value
    "${lhs};${elements}"                  # children
    )
  ans(ast)

  map_set(${ast} indexation_lhs ${lhs})
  map_set(${ast} indexation_elements ${elements})
  compile_indexation("${ast}")  
  return_ref(ast)
endfunction()

