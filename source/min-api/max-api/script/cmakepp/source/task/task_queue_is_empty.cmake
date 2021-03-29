## 
##
## returns true iff the task queue is empty
function(task_queue_is_empty task_queue)
  linked_list_peek_front("${task_queue}")
  ans(first)
  if(first)
    return(false)
  endif()
  return(true)
endfunction()
