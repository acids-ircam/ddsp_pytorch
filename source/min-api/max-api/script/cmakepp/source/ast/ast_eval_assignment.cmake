
  function(ast_eval_assignment ast scope)
    message("eval assignment")
    map_get(${ast} children)
    #ans(children)

    #address_get(${children})
    ans(rvalue)
    list_pop_front( rvalue)
    ans(lvalue)
    address_print("${lvalue}")
    address_print("${rvalue}")
    ast_eval(${rvalue} ${scope})
    ans(val)
    message("assigning value ${val} to")

    map_get(${lvalue} types)
    ans(types)
    message("types for lvalue ${types}")

    map_get(${lvalue} identifier)
    ans(identifier)
    map_set(${scope} "${identifier}" ${val})

  endfunction()