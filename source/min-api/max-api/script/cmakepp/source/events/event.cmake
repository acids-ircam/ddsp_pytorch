## `(<event-id>):<event>`
##
## tries to get the `<event>` identified by `<event-id>`
## if it does not exist a new `<event>` is created by  @markdown_see_function("event_new(...)")
function(event )
  set(event_id ${ARGN}) 
  set(event)
  if(event_id)
    event_get("${event_id}")
    ans(event)
  endif()
  if(NOT event)
    event_new(${event_id})
    ans(event)
  endif()
  return_ref(event)
endfunction()
