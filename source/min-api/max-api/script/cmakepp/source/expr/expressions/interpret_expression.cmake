
function(interpret_expression tokens)
  set(inner_exceptions)

  interpret_assign("${tokens}")
  ans(ast)
  if(ast)
    return(${ast})
  endif()
  ans_append(inner_exceptions)

  interpret_rvalue("${tokens}")
  ans(ast)
  if(ast)
    return(${ast})
  endif()
  ans_append(inner_exceptions)



  throw("could not intepret expression" ${inner_exceptions} --function interpret_expression)
endfunction()