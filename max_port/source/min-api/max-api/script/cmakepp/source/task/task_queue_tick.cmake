##
##
## executes the next task in the task queue 
## returns the executed task. 
## returns nothing if the task queue is empty
function(task_queue_tick task_queue)
  
  linked_list_pop_front("${task_queue}")
  ans(task)
  if(NOT task)
    return()
  endif()
  
  map_tryget("${task_queue}" can_tick)
  ans(can_tick)
  if(can_tick)
    call2("${can_tick}" "${task_queue}")
    ans(can_tick)    
    if(NOT can_tick)
      return()
    endif()
  endif()


  task_invoke("${task}")
  ans(success)

  return_ref(task)
endfunction()
