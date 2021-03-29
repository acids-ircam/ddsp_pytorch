
  function(token_stream_commit stream)
    map_get(${stream}  stack)
    ans(stack)
    stack_pop(${stack})
  endfunction()