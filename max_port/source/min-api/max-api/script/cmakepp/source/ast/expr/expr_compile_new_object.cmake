function(expr_compile_new_object)
  map_tryget(${ast}  children)
  ans(keyvalues)
  map_tryget(${keyvalues}  children)
  ans(keyvalues)

  make_symbol()
  ans(symbol)

  set(evaluation)
  foreach(keyvalue ${keyvalues})
    map_tryget(${keyvalue}  children)
    ans(pair)
    list_extract(pair key_ast value_ast)
    map_tryget(${key_ast}  data)
    ans(key)
    ast_eval(${value_ast} ${context})
    ans(value)
    #string(REPLACE "\${" "\${" value "${value}")
    set(evaluation "${evaluation}
    ${value}
    ans(${symbol}_tmp)
    map_set(\"\${${symbol}}\" \"${key}\" \"\${${symbol}_tmp}\")")
  endforeach()

  set(res "
  #expr_compile_new_object
  map_new()
  ans(${symbol})
  ${evaluation}
  set_ans_ref(${symbol})
  #end of expr_compile_new_object
  ")

  return_ref(res)

endfunction()