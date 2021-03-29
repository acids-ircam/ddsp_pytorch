function(expr_compile_expression)
  #message("compiling expression")
  map_get(${ast}  children)
  ans(children)
  set(result "")
  
  list(LENGTH children len)
  if(len GREATER 1)

    make_symbol()
    ans(symbol)
    foreach(rvalue_ast ${children})
      ast_eval(${rvalue_ast} ${context})
      ans(rvalue)

      set(result "${result}
  ${rvalue}
  ans(left)")
      map_set(${context} left ${rvalue})
      map_set(${context} left_ast ${rvalue_ast})
    endforeach()
    
    map_append_string(${context} code "
#expr_compile_expression
function(${symbol})
  set(left)
  ${result}
  return_ref(left)
endfunction()
#end of expr_compile_expression")

    set(symbol "
  #expr_compile_expression
  ${symbol}()
  #end of expr_compile_expression")
  else()
    ast_eval(${children} ${context})
    ans(symbol)
  endif()


  return_ref(symbol)
endfunction()