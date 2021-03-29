
function(interpret_key_value tokens)
  set(rhs_tokens)
  set(colon_token)
  set(key_tokens ${tokens})
  while(key_tokens)
    list_pop_back(key_tokens)
    ans(token)

    map_tryget("${token}" type)
    ans(token_type)

    if("${token_type}" STREQUAL "colon")
      set(colon_token "${token}")
      break()
    endif()

    list(APPEND rhs_tokens ${token})
  endwhile()

  if(NOT colon_token)
    throw("missing colon" --function interpret_key_value)
  endif()

  if(NOT rhs_tokens)
    throw("missing tokens for the value" --function interpret_key_value)
  endif()

  if(NOT key_tokens)
    throw("no key specified" --function interpret_key_value)
  endif()

  interpret_rvalue("${rhs_tokens}")
  rethrow()
  ans(rhs)

  interpret_rvalue("${key_tokens}")
  rethrow()
  ans(key)



  ast_new(
    "${tokens}"         # tokens
    "key_value"         # expression_type
    "key_value"         # value_type
    ""                  # ref
    ""                  # code
    ""                  # value
    "false"             # const
    "false"             # pure_value
    "${key};${rhs}"     # children
    )
  ans(ast)

  map_set("${ast}" key_ast ${key})
  map_set("${ast}" value_ast ${rhs})

  return_ref(ast)
endfunction()
