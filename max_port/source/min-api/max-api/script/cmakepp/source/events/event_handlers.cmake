## `(<event>)-><event handler...>`
##
## returns all handlers registered for the event
function(event_handlers event)
  event_get("${event}")
  ans(event)

  if(NOT event)
    return()
  endif()

  map_tryget(${event} handlers)
  return_ans()

endfunction()