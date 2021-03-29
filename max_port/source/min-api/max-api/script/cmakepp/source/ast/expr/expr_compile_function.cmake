function(expr_compile_function) # context, ast
 # message("expr_compile_function")

  map_tryget(${ast} children)
  ans(children)

  #message("children ${children}")

  list_extract(children signature_ast body_ast)

  map_tryget(${signature_ast} children)
  ans(signature_identifiers)
  set(signature_vars)
  set(identifiers)
  foreach(identifier ${signature_identifiers})
    map_tryget(${identifier} data)
    ans(identifier)
    list(APPEND identifiers "${identifier}")
    set(signature_vars "${signature_vars} ${identifier}")
  endforeach()  
  #message("signature_identifiers ${identifiers}")

  map_tryget(${body_ast} types)
  ans(body_types)

  list_contains(body_types closure)
  ans(is_closure)
  
  if(is_closure)
   map_tryget(${body_ast} children)
    ans(body_ast)

  endif()

  make_symbol()
  ans(symbol)
 # message("body_types ${body_types}")

  ast_eval(${body_ast} ${context})
  ans(body)

map_append_string(${context} code "#expr_compile_function
function(\"${symbol}\"${signature_vars})
  map_new()
  ans(local)  
  map_capture(\"\${local}\" this global${signature_vars})
  ${body}
  return_ans()
endfunction()
#end of expr_compile_function")
  

  set(res "set_ans(\"${symbol}\")")

  return_ref(res)  
endfunction()