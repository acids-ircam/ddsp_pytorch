function(expr_compile_number) # scope, ast

  map_tryget(${ast}  data)
  ans(data)
  make_symbol()
  ans(symbol)
  
 
  set(res "
  # expr_compile_number
  set_ans(\"${data}\")
  # end of expr_compile_number")
  return_ref(res)  
endfunction()