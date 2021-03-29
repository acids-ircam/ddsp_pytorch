# process_kill(<process handle?!>)
# stops the process specified by <process handle?!>
# returns true if the process was killed successfully
function(process_kill)
  wrap_platform_specific_function(process_kill)
  process_kill(${ARGN})
  return_ans()
endfunction()
