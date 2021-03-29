## returns the runtime unique process handle
## information may differ depending on os but the following are the same for any os
## * pid
## * status
function(process_handle handlish)
  is_map("${handlish}")
  ans(ismap)

  if(ismap)
    set(handle ${handlish})
  elseif( "${handlish}" MATCHES "[0-9]+")
    string(REGEX MATCH "[0-9]+" handlish "${handlish}")

    map_tryget(__process_handles ${handlish})
    ans(handle)
    if(NOT handle)
      map_new()
      ans(handle)
      map_set(${handle} pid "${handlish}")          
      map_set(${handle} state "unknown")
      map_set(__process_handles ${handlish} ${handle})
    endif()
  else()
    message(FATAL_ERROR "'${handlish}' is not a valid <process handle>")
  endif()
  return_ref(handle)
endfunction()
