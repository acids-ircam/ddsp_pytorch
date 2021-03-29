
  function(ast_json_eval_array )#ast scope
    map_get(${ast}  children)
    ans(values)
    set(res)
    foreach(value ${values})
      ast_eval(${value} ${context})
      ans(evaluated_value)
      list(APPEND res "${evaluated_value}")
    endforeach()
    return_ref(res)
  endfunction()