
function(interpret_scope_rvalue tokens)
  #print_vars(tokens)
  list(LENGTH tokens count)
  if(NOT ${count} EQUAL 2)
    throw("expected 2 tokens (got {count}) " --function interpret_scope_rvalue)    
  endif()

  list(GET tokens 0 dollar)
  list(GET tokens 1 identifier_token)


  map_tryget("${dollar}" type)
  ans(type)
  if(NOT "${type}" STREQUAL "dollar")
    throw("expected a `$` as first token " --function interpret_scope_rvalue)
  endif()


  if(NOT identifier_token)
    throw("could find identifier" --function interpret_scope_rvalue)
  endif()


  map_tryget("${identifier_token}" type)
  ans(type)

  set(identifier)
  if("${type}" MATCHES "^(quoted)|(unquoted)$")
    interpret_literal("${identifier_token}")
    ans(identifier)
  elseif("${type}" STREQUAL "paren")
    interpret_paren("${identifier_token}" )
    ans(identifier)
  elseif("${type}" STREQUAL "bracket")
    interpret_list("${identifier_token}")
    ans(identifier)
  endif()
  if(NOT identifier)
    throw("could interpret identifier" --function interpret_scope_rvalue)
  endif()


  map_tryget("${identifier}" code)
  ans(identifier_code)

  map_tryget("${identifier}" value)
  ans(value)

  map_tryget("${identifier}" pure_value)
  ans(pure_value)

  set(code)

  ast_new(
    "${tokens}"
    scope_rvalue
    any
    "${value}"
    "${code}"
    "\${${value}}" # value
    "false" # const
    "${pure_value}"
    "${identifier}" # children

    )
  
  ans(ast)

  return_ref(ast)
endfunction()