function(interpret_navigation_rvalue tokens)
  list(LENGTH tokens token_count)
  if(${token_count} LESS 2)
    throw("expected at least two tokens (got {token_count})" --function interpret_navigation_rvalue)
  endif()

  list_pop_back(tokens)
  ans(rhs_token)

  if(NOT tokens)
    throw("expected at least two tokens (got {token_count})" --function interpret_navigation_rvalue)
  endif()

  map_tryget("${rhs_token}" type)
  ans(type)

  list_pop_back(tokens)
  ans(dot)
  map_tryget("${dot}" type)
  ans(type)
  if(NOT "${type}" STREQUAL "dot")
    throw("expected second to last token to be a `.`" --function interpret_navigation_rvalue)
  endif()  
  
  if(NOT tokens)
    throw("no lvalue tokens" --function interpret_navigation_rvalue)
  endif()

  interpret_literal("${rhs_token}")
  ans(rhs)

  interpret_rvalue("${tokens}")
  ans(lhs)

  #print_vars(rhs.type rhs.argument rhs.code lhs.type lhs.argument lhs.code --plain)

  map_tryget("${rhs}"  value)
  ans(rhs_value)

  map_tryget("${lhs}"  value)
  ans(lhs_value)

  next_id()
  ans(ref)

  set(code "get_property(__ans GLOBAL PROPERTY \"${lhs_value}.__object__\" SET)
if(__ans)
  message(FATAL_ERROR object_get_not_supported_currently)
else()
  get_property(${ref} GLOBAL PROPERTY \"${lhs_value}.${rhs_value}\")
endif()    
")

  ast_new(
    "${tokens}"
    "navigation_rvalue"  
    "any"           # return type
    "${ref}"        # ref
    "${code}"       # code
    "\${${ref}}"    #value
    "false"         #const
    "false"
    "${rhs};${lhs}"
    )
  ans(ast)

  map_set("${ast}" this_ref "${lhs_argument}")

  return_ref(ast)
endfunction()