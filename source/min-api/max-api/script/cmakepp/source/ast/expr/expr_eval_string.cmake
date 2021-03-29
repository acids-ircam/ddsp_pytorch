function(expr_eval_string) # scope, ast
  map_tryget(${ast}  data)
  ans(data)
  return_ref(data)  
endfunction()