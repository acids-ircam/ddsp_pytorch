function(expr_compile_indexation)
  map_tryget(${ast}  children)
  ans(indexation_expression_ast)
  ast_eval(${indexation_expression_ast} ${context})
  ans(indexation_expression)

  set(res "
  # expr_compile_indexation
  ${indexation_expression}
  ans(index)
  set(this \"\${left}\")
  map_get(\"\${this}\" \"\${index}\")
  # end of expr_compile_indexation")


  return_ref(res)
endfunction()