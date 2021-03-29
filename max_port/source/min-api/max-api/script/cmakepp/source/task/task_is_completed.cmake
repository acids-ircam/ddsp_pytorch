## `(<task>)-><bool>`
##
## returns true if task is complete
function(task_is_completed task)
  task_state("${task}")
  ans(state)
  if("${state}" STREQUAL "completed")
    return(true)
  endif()
  return(false)
endfunction()
