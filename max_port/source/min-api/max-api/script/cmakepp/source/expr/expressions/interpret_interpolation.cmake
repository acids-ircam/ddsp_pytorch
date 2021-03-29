## parse elements with whitespace/newline as delimiter
function(interpret_interpolation tokens)
  if(NOT tokens)
    throw("expected at least one token" --function interpret_interpolation)
  endif()
  set(code)
  set(literals)
  set(value)

  foreach(token ${tokens})
    interpret_literal("${token}")
    ans(literal)
    if(NOT literal)
      ## something other than a literal 
      ## causes this not to be a literal
      throw("tokens contained an invalid token: {token}" --function interpret_interpolation)
    endif()

    #print_vars(literal.type literal.value literal.code)

    list(APPEND literals "${literal}")

    map_tryget("${literal}" code)
    ans(literal_code)
    map_tryget("${literal}" value)
    ans(literal_value)
    set(value "${value}${literal_value}")    
  endforeach()
 #   print_vars(value )

  ## single literal is retunred directly
  ## 
  list(LENGTH literals length)
  if("${length}" LESS 2)
    return_ref(literals)
  endif()


  next_id()
  ans(ref)
  
  set(code "set(${ref} \"${value}\")\n")


  ast_new(
    "${tokens}"
    literals            # expression type
    composite_string    # value type
    "${ref}"            # ref
    "${code}"           # code
    "${value}"          # value
    "true"              # const
    "true"              # pure value
    "${literals}"       # children
    )
  ans(ast)
  return_ref(ast)
endfunction()

