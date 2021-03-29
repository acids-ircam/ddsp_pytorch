function(expr_compile_list)
  map_tryget(${ast}  children) 
  ans(element_asts)
  set(arguments)
  set(evaluation)
  set(i 0)

  make_symbol()
  ans(symbol)
  set(elements)
  foreach(element_ast ${element_asts})
    ast_eval(${element_ast} ${context})
    ans(element)

    set(evaluation "${evaluation}
  ${element}
  ans(${symbol}_arg${i})")
    set(elements "${elements}\"\${${symbol}_arg${i}}\" " )
    math(EXPR i "${i} + 1")
  endforeach()
  set(res "
  #expr_compile_list
  ${evaluation}
  set(${symbol} ${elements})
  set_ans_ref(${symbol})
  #end of expr_compile_list")
  return_ref(res)
endfunction()