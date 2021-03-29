function(expr_compile_expression_statement) # context, ast
  map_tryget(${ast}  children)
  ans(statement_ast)
  ast_eval(${statement_ast} ${context})
  ans(statement)
  set(res "
  # expr_compile_statement
  ${statement}
  # end of expr_compile_statement")
  return_ref(res)  
endfunction()