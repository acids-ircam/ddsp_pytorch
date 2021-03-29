## process_info implementation for linux_ps
## currently only returns the process command name
function(process_info_Linux handle)
  process_handle("${handle}")
  ans(handle)

  map_tryget(${handle} pid)
  ans(pid)
  

  linux_ps_info_capture(${pid} ${handle} comm)


  return_ref(handle)    
endfunction()

