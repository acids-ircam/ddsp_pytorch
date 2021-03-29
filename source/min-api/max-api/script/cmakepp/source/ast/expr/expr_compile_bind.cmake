function(expr_compile_bind)
  set(res "
  # expr_compile_bind 
  set(this \"\${left}\")
  # end of expr_compile_bind")
  return_ref(res)
endfunction()