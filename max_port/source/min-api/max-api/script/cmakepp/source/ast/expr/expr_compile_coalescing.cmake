function(expr_compile_coalescing)
  map_tryget(${ast}  children)
  ans(expr_ast)
  ast_eval(${expr_ast} ${context})
  ans(expr)
  set(res "
  # expr_compile_coalescing 
  if(NOT left)
    ${expr}
  endif()
  # end of expr_compile_coalescing")
  return_ref(res)
endfunction()