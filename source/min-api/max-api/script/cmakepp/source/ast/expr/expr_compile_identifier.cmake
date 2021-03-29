function(expr_compile_identifier)# ast context
  
#message("ast: ${ast}")
  
  map_tryget(${ast}  data)
  ans(data)
  set(res "
  # expr_compile_identifier
  #map_tryget(\"\${local}\" \"${data}\")
  scope_resolve(\"${data}\")
  obj_get(\"\${this}\" \"${data}\")
  # end of expr_compile_identifier")
  return_ref(res)
endfunction()