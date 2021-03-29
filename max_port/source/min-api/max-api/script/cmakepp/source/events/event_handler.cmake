## `(<~callable>)-><event handler>` 
##
## creates an <event handler> from the specified callable
## and returns it. a `event_handler` is also a callable
function(event_handler callable)
  callable("${callable}")
  ans(event_handler)
  return_ref(event_handler)
endfunction()
