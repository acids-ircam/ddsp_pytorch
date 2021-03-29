## `(<event-id...>)-><event tracker>`
##
## sets up a function which listens only to the specified events
## 
function(events_track)
  function_new()
  ans(function_name)

  map_new()
  ans(map)

  eval("
    function(${function_name})
      map_new()
      ans(event_args)
      map_tryget(\${event} event_id)
      ans(event_id)
      map_set(\${event_args} id \${event_id})
      map_set(\${event_args} args \${ARGN})
      map_set(\${event_args} event \${event})
      map_append(${map} \${event_id} \${event_args})
      map_append(${map} event_ids \${event_id})
      return(\${event_args})
    endfunction()
  ")

  foreach(event ${ARGN})
    event_addhandler(${event} ${function_name})
  endforeach()

  return(${map})
endfunction()