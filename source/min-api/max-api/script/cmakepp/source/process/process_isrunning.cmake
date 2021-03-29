## returns true iff the process identified by <handlish> is running
function(process_isrunning)    
  wrap_platform_specific_function(process_isrunning)    
  process_isrunning(${ARGN})
  return_ans()
endfunction()

