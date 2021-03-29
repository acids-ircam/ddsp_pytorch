## `(<~event>)-><void>`
##
## removes all handlers from the specified event
function(event_clear event)
  event_get("${event}")
  ans(event)

  event_handlers("${event}")
  ans(handlers)

  foreach(handler ${handlers})
    event_removehandler("${event}" "${handler}")
  endforeach()  

  return()
endfunction()

