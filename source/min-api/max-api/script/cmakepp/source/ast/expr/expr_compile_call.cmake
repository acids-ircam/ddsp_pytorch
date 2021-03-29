function(expr_compile_call)
  map_tryget(${ast}  children) 
  ans(argument_asts)
  set(arguments)
  set(evaluation)
  set(i 0)


  make_symbol()
  ans(symbol)

  foreach(argument_ast ${argument_asts})
    ast_eval(${argument_ast} ${context})
    ans(argument)

    set(evaluation "${evaluation}
  ${argument}
  ans(${symbol}_arg${i})")
    set(arguments "${arguments}\"\${${symbol}_arg${i}}\" " )
    math(EXPR i "${i} + 1")
  endforeach()

  set(res "
  # expr_compile_call 
  ${evaluation}
  call(\"\${left}\"(${arguments}))
  # end of expr_compile_call")

  return_ref(res)
endfunction()