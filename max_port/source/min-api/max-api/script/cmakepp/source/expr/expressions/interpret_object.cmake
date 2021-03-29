
function(interpret_object brace)
  list(LENGTH brace token_count)
  if(NOT "${token_count}" EQUAL 1)
    throw("expected single token, got {token_count}")
  endif()

  map_tryget("${brace}" type)
  ans(brace_token_type)

  if(NOT "${brace_token_type}" STREQUAL "brace")
    throw("expected brace token, got {brace_token_type}")
  endif()

  map_tryget("${brace}" tokens)
  ans(member_tokens)

  interpret_elements("${member_tokens}" "comma" "interpret_key_value;interpret_rvalue")
  ans(members)



  next_id()
  ans(ref)
  set(value "\${${ref}}")
  set(code "map_new()\nans(${ref})\n")
  foreach(member ${members})
    map_tryget("${member}" expression_type)
    ans(member_type)
    if("${member_type}" STREQUAL "key_value")
      map_tryget("${member}" key_ast)
      ans(member_key_ast)
      map_tryget("${member_key_ast}" value )
      ans(member_key_ast_value)
      map_tryget("${member}" value_ast)
      ans(member_value_ast)
      map_tryget("${member_value_ast}" value)
      ans(member_value_ast_value)
      set(code "${code}map_set(${value} \"${member_key_ast_value}\" \"${member_value_ast_value}\")\n")
    else() ## normal rvalue
      map_tryget("${member}" value)
      ans(member_value)
      set(code "${code}address_append(${value} \"${member_value}\")\n")
    endif()
  endforeach()


  ast_new(
    "${brace}"          # tokens
    "object"            # expression_type
    "object"            # value_type
    "${ref}"            # ref
    "${code}"           # code
    "${value}"          # value
    "false"             # const
    "false"             # pure_value
    "${members}"        # children
    )
  ans(ast)
  return_ref(ast)

endfunction()

