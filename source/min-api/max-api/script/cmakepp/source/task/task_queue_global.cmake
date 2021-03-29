##
##
## returns the global task_queue
function(task_queue_global)
  task_queue_new()
  ans(task_queue)
  eval_cmake(
    function(task_queue_global)
      return(${task_queue})
    endfunction()
    )
  return(${task_queue})
endfunction()
