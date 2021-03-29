function(interpret_navigation_lvalue tokens)
  list(LENGTH tokens token_count)
  if(${token_count} LESS 2)
    throw("expected at least two tokens (got {token_count})" --function interpret_navigation_rvalue)
  endif()

  set(lhs_tokens ${tokens})

  list_pop_back(lhs_tokens)
  ans(rhs_token)

  list_pop_back(lhs_tokens)
  ans(dot)

  map_tryget("${dot}" type)
  ans(type)
  if(NOT "${type}" STREQUAL "dot")
    throw("expected second to last token to be a `.`" --function interpret_navigation_rvalue)
  endif()  
  
  if(NOT lhs_tokens)
    throw("no lvalue tokens" --function interpret_navigation_rvalue)
  endif()

  interpret_literal("${rhs_token}")
  ans(rhs)

  map_tryget("${rhs}" value)
  ans(rhs_value)


  interpret_rvalue("${lhs_tokens}")
  ans(lhs)


  map_tryget("${lhs}" expression_type)
  ans(lhs_type)



  map_tryget("${lhs}" value)
  ans(lhs_value)

  next_id()
  ans(ref)

  set(pre_code)
  set(code "map_tryget(${lhs_value} \"${rhs_value}\")\nans(${ref})\n")
  if("${lhs_type}" STREQUAL ellipsis)
    set(post_code "foreach(v ${lhs_value})\nmap_set(\"\${v}\" \"${rhs_value}\" \${${ref}})\nendforeach()\n")
  else()
    set(post_code "map_set(${lhs_value} \"${rhs_value}\" \${${ref}})\n")
  endif()


  ast_new(
    "${tokens}"  # tokens 
    "navigation_lvalue"      # expression_type 
    "any"         # value_type 
    "${ref}"      # ref 
    "${code}"     # code
    "\${${ref}}"  # value 
    "false"      # const 
    "false"          # pure_value
    "${rhs};${lhs}"    # children
    )
  ans(ast)
  map_set("${ast}" pre_code "${pre_code}")
  map_set("${ast}" post_code "${post_code}")
  return_ref(ast)
endfunction()