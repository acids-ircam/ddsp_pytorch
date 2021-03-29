## `(<any>)-><bool>`
##
## returns true if the specified value is an event
## an event is a ref which is callable and has an event_id
##
function(is_event event)
  is_address("${event}")
  ans(is_ref)
  if(NOT is_ref)
    return()
  endif()
  is_callable("${event}")
  ans(is_callable)
  if(NOT is_callable)
    return(false)
  endif()

  map_has(${event} event_id)
  ans(has_event_id)
  if(NOT has_event_id)
    return(false)
  endif()

  return(true)
endfunction()