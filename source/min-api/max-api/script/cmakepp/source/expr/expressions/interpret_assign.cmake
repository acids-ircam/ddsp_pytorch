function(interpret_assign tokens)
  set(rhs_tokens)
  set(lhs_tokens ${tokens})
  set(equals_token)

  while(lhs_tokens)
    list_pop_back(lhs_tokens)
    ans(token)
    map_tryget("${token}" type)
    ans(token_type)
    if("${token_type}" STREQUAL "equals")
      set(equals_token "${token}")
      break()
    endif()
    list(APPEND rhs_tokens ${token})
  endwhile()

  if(NOT equals_token)
    throw("missing equals token" --function interpret_assign)
  endif()

  if(NOT lhs_tokens)
    throw("missing left hand side" --function interpret_assign)
  endif()

  if(NOT rhs_tokens)
    throw("missing right hand side" --function interpret_assign)
  endif()

  interpret_rvalue("${rhs_tokens}")
  rethrow()
  ans(rhs)

  map_tryget("${rhs}" value)
  ans(rhs_value)

  interpret_lvalue("${lhs_tokens}")
  rethrow()
  ans(lhs)


  map_tryget("${lhs}" value)
  ans(value)

  map_tryget("${lhs}" ref)
  ans(ref)


  map_tryget("${lhs}" value_type)
  ans(value_type)

  map_tryget("${lhs}" const)
  ans(const)


  set(code "set(${ref} ${rhs_value})\n")
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
    "assign"
    "${value_type}"
    "${ref}"
    "${code}"
    "${value}"
    "${const}"
    "false"
    "${rhs};${lhs}"
    )
  ans(ast)

  return_ref(ast)
endfunction()


