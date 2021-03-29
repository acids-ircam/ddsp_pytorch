## 
##
## creates a new task queue which is a linked list of tasks.
function(task_queue_new)
  linked_list_new()
  ans(task_queue)
  map_set_special(${task_queue} $type task_queue)
  map_set(${task_queue} is_running false)
  map_set(${task_queue} ticks 0)
  return(${task_queue})
endfunction()
