
## platform specific implementation for process_isrunning under windows
function(process_isrunning_Windows handlish)    
  process_handle("${handlish}")    
  ans(handle)    
  map_tryget(${handle} state)
  ans(state)
  if("${state}_" STREQUAL "terminated_" )
    return(false)
  endif()

  map_tryget(${handle} pid)
  ans(pid)
  
  win32_tasklist_bare(-FI "PID eq ${pid}" -FI "STATUS eq Running")
  ans_extract(error)
  ans(res)
  if("${res}" MATCHES "${pid}")
    return(true)
  endif()
  return(false)
endfunction()

