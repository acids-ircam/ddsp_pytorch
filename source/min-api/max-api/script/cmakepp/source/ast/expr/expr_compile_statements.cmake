function(expr_compile_statements) # scope, ast
  map_tryget(${ast}  children)
  ans(statement_asts)
  set(statements)
  #message("children: ${statement_asts}")
  list(LENGTH statement_asts len)
  set(index 0)
  foreach(statement_ast ${statement_asts})
    math(EXPR index "${index} + 1")
    ast_eval(${statement_ast} ${context})
    ans(statement)
    set(statements "${statements}
  #statement ${index} / ${len}
  ${statement}")
  endforeach()
  map_tryget(${ast}  data)
  ans(data)
  make_symbol()
  ans(symbol)
  
  make_symbol()
  ans(symbol)

  map_append_string(${context} code "
# expr_compile_statements
function(\"${symbol}\")
  ${statements}
  return_ans()
endfunction()
# end of expr_compile_statements")
  
  set(res "${symbol}()")

#  message("${res}")
  return_ref(res)  
endfunction()