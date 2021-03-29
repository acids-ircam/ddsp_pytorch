## `(<~event>)-><event>`
##  
## returns the `<event>` identified by `<event-id>` 
## if the event does not exist `<null>` is returned.
function(event_get event)
  events()
  ans(events)

  is_event("${event}")
  ans(is_event)

  if(is_event)
    return_ref(event)
  endif()
  
  map_tryget(${events} "${event}")
  return_ans()
endfunction()
