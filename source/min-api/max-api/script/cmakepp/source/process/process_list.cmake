## returns a list of <process info> containing all processes currently running on os
## process_list():<process info>...
function(process_list)
  wrap_platform_specific_function(process_list)
  process_list(${ARGN})
  return_ans()
endfunction()
