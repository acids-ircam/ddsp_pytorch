
  function(ast_json_eval_number )#ast scope
    map_get(${ast}  data)
    ans(data)
    return_ref(data)
  endfunction()