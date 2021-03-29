function(expr_compile_new)
#json_print(${ast})
  map_tryget(${ast} children)
  ans(children)

  list_extract(children className_ast call_ast)

  map_tryget(${className_ast} data)
  ans(className)

  map_tryget(${call_ast} children)
  ans(argument_asts)


 # message("class name is ${className} ")

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
#expr_compile_new
${evaluation}
obj_new(\"${className}\" ${arguments})
#end of expr_compile_new
  ")


return_ref(res)
endfunction()