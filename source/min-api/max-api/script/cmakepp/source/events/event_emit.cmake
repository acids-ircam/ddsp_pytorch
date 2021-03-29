## `(<~event> <args:<any...>>)-><any...>`
##
## emits the specified event. goes throug all event handlers registered to
## this event and 
## if event handlers are added during an event they will be called as well
##
## if a event calls event_cancel() 
## all further event handlers are disregarded
##
## returns the accumulated result of the single event handlers
function(event_emit event)
  is_event("${event}")
  ans(is_event)
  
  if(NOT is_event)
    event_get("${event}")
    ans(event)
  endif()


  if(NOT event)
    return()
  endif()


  set(result)

  set(previous_handlers)
  # loop aslong as new event handlers are appearing
  # 
  address_new()
  ans(__current_event_cancel)
  address_set(${__current_event_cancel} false)
  while(true)
    ## 
    map_tryget(${event} handlers)
    ans(handlers)
    list_remove(handlers ${previous_handlers} "")
    list(APPEND previous_handlers ${handlers})

    list_length(handlers)
    ans(length)
    if(NOT "${length}" GREATER 0) 
      break()
    endif()

    foreach(handler ${handlers})

      event_handler_call(${event} ${handler} ${ARGN})
      ans(success)
      list(APPEND result "${success}")
      ## check if cancel is requested
      address_get(${__current_event_cancel})
      ans(break)
      if(break)
        return_ref(result)
      endif()
    endforeach()
  endwhile()

  return_ref(result)
endfunction() 
