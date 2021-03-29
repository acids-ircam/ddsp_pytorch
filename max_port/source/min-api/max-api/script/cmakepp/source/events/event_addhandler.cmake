## `event_addhandler(<~event> <~callable>)-><event handler>`
##
## adds an event handler to the specified event. returns an `<event handler>`
## which can be used to remove the handler from the event.
##
function(event_addhandler event handler)
  event("${event}")
  ans(event)

  event_handler("${handler}")
  ans(handler)

  ## then only append function 
  map_append_unique("${event}" handlers "${handler}")
 
  return(${handler})  
endfunction()

