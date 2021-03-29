##
##
## pushes a task onto the back of the task queue
function(task_queue_push task_queue task)
  task("${task}" ${ARGN})
  ans(task)
  map_set("${task}" task_queue "${task_queue}")
  linked_list_push_back("${task_queue}" ${task})
  ans(node)
  return(${task})
endfunction()
