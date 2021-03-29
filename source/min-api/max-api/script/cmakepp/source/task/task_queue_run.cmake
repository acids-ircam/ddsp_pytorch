## 
##
## runs the task queue until it is either empty, maximum number of ticks reached or times out
## returns the number of tasks that where run
## at least one task is run
function(task_queue_run task_queue)
  set(args ${ARGN})
  list_extract_labelled_value(args --ticks)
  ans(max_ticks)
  list_extract_labelled_value(args --timeout)
  ans(timeout)
  set(timer)
  if(NOT "${timeout}_" STREQUAL "_")
    address_new()
    ans(timer)
  endif()

  if("${max_ticks}_" STREQUAL "_")
    set(max_ticks -1)
  endif()

  if(timer)
    timer_start(${timer})
  endif()

  set(ticks 0)
  while(true)
    task_queue_tick("${task_queue}")
    ans(completed_task)
    if(NOT completed_task)
      break()
    endif()
    math(EXPR ticks "${ticks} + 1")
    if("${max_ticks}" GREATER -1 AND NOT ${ticks} LESS ${max_ticks})
      break()
    endif()
    if(timer)
      timer_elapsed(${timer})
      ans(elapsed_time)
      if("${elapsed_time}" GREATER "${timeout}")
        break()
      endif()
    endif()
  endwhile()

  return(${ticks})
endfunction()