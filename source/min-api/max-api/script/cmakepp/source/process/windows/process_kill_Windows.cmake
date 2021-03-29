
# windows implementation for process kill
function(process_kill_Windows process_handle)
  process_handle("${process_handle}")
  map_tryget(${process_handle} pid)
  ans(pid)

  win32_taskkill(/PID ${pid} --exit-code)
  ans(exit_code)
  if(exit_code)
    return(false)
  endif()
  return(true)
endfunction()
