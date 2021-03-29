
  function(ast_json_eval_key_value )#ast scope
    map_get(${ast}  children)
    ans(value)
    list_pop_front( value)
    ans(key)
    ast_eval(${key} ${context})
    ans(key)
    ast_eval(${value} ${context})
    ans(value)

    #message("keyvalue ${key}:${value}")
    map_set(${context} ${key} ${value})
  endfunction()