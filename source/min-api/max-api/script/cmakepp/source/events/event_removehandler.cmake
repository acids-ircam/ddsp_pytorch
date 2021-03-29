## `(<event handler>)-><bool>`
##
## removes the specified handler from the event idenfied by event_id
## returns true if the handler was removed
function(event_removehandler event handler)

  event("${event}")
  ans(event)
  
  if(NOT event)
    return(false)
  endif()


  event_handler("${handler}")
  ans(handler)


  map_remove_item("${event}" handlers "${handler}")
  ans(success)
  
  return_truth("${success}")
  
endfunction()

