## `(<event> <event handler>)-><any>`
##
## calls the specified event handler for the specified event.
function(event_handler_call event event_handler)
  callable_call("${event_handler}" ${ARGN})
  ans(res)
  return_ref(res)
endfunction()
