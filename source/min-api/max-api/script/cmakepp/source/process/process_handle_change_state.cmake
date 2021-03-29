
function(process_handle_change_state process_handle new_state)
  map_tryget("${process_handle}" state)
  ans(old_state)
  if("${old_state}" STREQUAL "${new_state}")
    return(false)
  endif()

  map_tryget(${process_handle} on_state_change)
  ans(on_state_change_event)

  event_emit(${on_state_change_event} ${process_handle})



  map_set(${process_handle} state "${new_state}")

  if("${new_state}" STREQUAL "terminated")
    map_tryget(${process_handle} exit_code)
    ans(error)
    if(error)
      map_tryget("${process_handle}" on_error)
      ans(on_error_event)
      event_emit("${on_error_event}" ${process_handle})  
    else()
      map_tryget("${process_handle}" on_success)
      ans(on_success_event)
      event_emit("${on_success_event}" ${process_handle})  
    endif()
    map_tryget("${process_handle}" on_terminated)
    ans(on_terminated_event)
    event_emit("${on_terminated_event}" ${process_handle})
endif()

  return(true)
endfunction()