
## maybe use some kind of quick heuristic?
function(interpret_rvalue tokens)
  set(inner_exceptions)
  interpret_paren("${tokens}")
  ans(ast)
  if(ast)
    return(${ast})
  endif()
  ans_append(inner_exceptions)

  interpret_rvalue_dereference("${tokens}")
  ans(ast)
  if(ast)
    return(${ast})
  endif()
  ans_append(inner_exceptions)

  interpret_rvalue_reference("${tokens}")
  ans(ast)
  if(ast)
    return(${ast})
  endif()
  ans_append(inner_exceptions)

  interpret_default_value("${tokens}")
  ans(ast)
  if(ast)
    return(${ast})
  endif()
  ans_append(inner_exceptions)

  interpret_ellipsis("${tokens}")
  ans(ast)
  if(ast)
    return(${ast})
  endif()
  ans_append(inner_exceptions)



  interpret_literal("${tokens}")
  ans(ast)
  if(ast)
    return(${ast})
  endif()
  ans_append(inner_exceptions)

  interpret_interpolation("${tokens}")
  ans(ast)
  if(ast)
    return(${ast})
  endif()
  ans_append(inner_exceptions)
  
  interpret_bind_call("${tokens}")
  ans(ast)
  if(ast)
    return(${ast})
  endif()
  ans_append(inner_exceptions)

  interpret_call("${tokens}")
  ans(ast)
  if(ast)
    return(${ast})
  endif()
  ans_append(inner_exceptions)

  interpret_list("${tokens}")
  ans(ast)
  if(ast)
    return(${ast})
  endif()
  ans_append(inner_exceptions)

  interpret_object("${tokens}")
  ans(ast)
  if(ast)
    return(${ast})
  endif()
  ans_append(inner_exceptions)


  ## needs to come before navigation rvalue because $ needs to bind
  ## stronger
  interpret_scope_rvalue("${tokens}")
  ans(ast)
  if(ast)
    return(${ast})
  endif()
  ans_append(inner_exceptions)

  interpret_indexation("${tokens}")
  ans(ast)
  if(ast)
    return(${ast})
  endif()
  ans_append(inner_exceptions)

  interpret_navigation_rvalue("${tokens}")
  ans(ast)
  if(ast)
    return(${ast})
  endif()
  ans_append(inner_exceptions)




  throw("could not interpret rvalue" ${inner_exceptions} --function interpret_rvalue )
endfunction()



