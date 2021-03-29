## interpret an lvalue
## needs a rvalue
function(interpret_scope_lvalue tokens)
  
  if(NOT tokens)
    throw("missing tokens" --function interpret_scope_lvalue)
  endif()

  list(GET tokens 0 dollar_token)
  map_tryget("${dollar_token}" type)
  ans(dollar_token_type)


  if(NOT "${dollar_token_type}" STREQUAL "dollar")
    throw("first token is not the dollar token" --function interpret_scope_lvalue)
  endif()

  set(identifier_token ${tokens})
  list(REMOVE_AT identifier_token 0)
  list(LENGTH identifier_token identifier_token_count)
  if(NOT "${identifier_token_count}" EQUAL 1)
    throw("expected one identifier token got ${identifier_token_count}" --function interpret_scope_lvalue)
  endif()

  
  interpret_rvalue("${identifier_token}")
  rethrow()
  ans(identifier)

  map_tryget("${identifier}" expression_type)
  ans(expression_type)
  if(NOT "${expression_type}" MATCHES "(literal)|(paren)|(bracket)")
    throw("invalid identifier : expected a literal|paren|bracket as an identifier but got ${expression_type}")
  endif()

  map_tryget("${identifier}" value)
  ans(identifier_value)


  set(code)

  ## set both in scope and in parent scope
  set(post_code "set(\"${identifier_value}\" \${${identifier_value}} PARENT_SCOPE)\n")

  # tokens 
  # expression_type 
  # value_type 
  # ref 
  # code
  # value 
  # const 
  # pure_value
  # children


  ast_new(
    "${tokens}"
    "scope_lvalue"
    "any"
    "${identifier_value}"
    "${code}"
    "\${${identifier_value}}"
    "false"
    "false"
    "${identifier}"
    )
  ans(ast)
  map_set("${ast}" post_code "${post_code}")
  return_ref(ast)
endfunction()
