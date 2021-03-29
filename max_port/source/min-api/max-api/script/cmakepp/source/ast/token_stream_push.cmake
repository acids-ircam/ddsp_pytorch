
  function(token_stream_push stream)
    map_get(${stream}  stack)
    ans(stack)
    map_tryget(${stream}  current)
    ans(current)
    stack_push(${stack} ${current})

   # message("pushed")
  endfunction()