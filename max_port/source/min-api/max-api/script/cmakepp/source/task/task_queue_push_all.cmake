##
##
## pushes all specified tasks onto the task queue
function(task_queue_push_all task_queue)
  foreach(task ${ARGN})
    task_queue_push("${task_queue}" ${task})
  endforeach()

endfunction()