
function(interpret_call tokens)
  set(callable_tokens ${tokens})
  ## no tokens
  if(NOT callable_tokens)
    throw("no tokens specified" --function interpret_call)
  endif()


  list_pop_back(callable_tokens)
  ans(paren_token)

  ## call must end with a invocation "("  ")"
  map_tryget("${paren_token}" type)
  ans(type)
  if(NOT "${type}" STREQUAL "paren")
    throw("no value for left hand side" --function interpret_call)
  endif()

  ## call must have callable
  if(NOT callable_tokens)
    throw("no value for left hand side" --function interpret_call)
  endif()

  interpret_rvalue("${callable_tokens}")
  ans(callable_ast)
  if(NOT callable_ast)
    throw("could not parse rvalue" --function interpret_call)
  endif()

  ## get parameters
  map_tryget(${paren_token} tokens)
  ans(parameter_tokens)

  interpret_elements("${parameter_tokens}" "comma" "interpret_ellipsis;interpret_reference_parameter;interpret_expression")
  ans(parameter_asts) 

  ## create code for calling the function
  next_id()
  ans(ref)
  interpret_call_create_code("${ref}" "${callable_ast}" "${parameter_asts}")
  ans(code)

  ## return the ast
  ast_new(
    "${tokens}"
    call
    any
    "${ref}"
    "${code}"
    "\${${ref}}"
    false
    false
    "${rvalue};${parameter_asts}"
    )
  ans(ast)


  return_ref(ast)
endfunction()





