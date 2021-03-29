function(task_start task)
  task_queue_global()
  ans(task_queue)
  task_queue_push("${task_queue}" "${task}")
  return()
endfunction()

