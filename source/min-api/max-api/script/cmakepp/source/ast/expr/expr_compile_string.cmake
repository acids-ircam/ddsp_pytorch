function(expr_compile_string) # scope, ast

  map_tryget(${ast}  data)
  ans(data)
  make_symbol()
  ans(symbol)
  
 
  set(res "
  # expr_compile_string
  set_ans(\"${data}\")
  # end of expr_compile_string")
  return_ref(res)  
endfunction()