
  function(stack_new)
    address_new(stack)
    ans(stack)   
    map_set_hidden("${stack}" front 0)
    map_set_hidden("${stack}" back 0)
    return(${stack})
  endfunction()