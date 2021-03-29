## process_info(<process handle?!>): <process info>
## returns information on the specified process handle
function(process_info)
  wrap_platform_specific_function(process_info)
  process_info(${ARGN})
  ans(res)
  return_ref(res)
endfunction()
