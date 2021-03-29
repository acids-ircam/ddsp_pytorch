
  function(token_stream_move_next stream)
    map_get(${stream}  current)
    ans(current)
    map_tryget(${current}  next)
    ans(next)
    map_set(${stream} current ${next})
   # message(FORMAT "moved from {current.data} to {next.data}")
  endfunction()