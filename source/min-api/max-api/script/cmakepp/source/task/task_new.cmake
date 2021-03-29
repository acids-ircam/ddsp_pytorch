## `(<~callable> <any>...)-><task>`
## 
## creates a new task accepting a callable and arguments vor its invocation
##
## ```
## <task> ::= {
##   callable: <callable>   ## contains the callable that is invoked
##   arguments: <any>...    ## contains the arguments that the task is invoked with
##   return_value: <any>    ## is set when task is complete
## }
## ```
function(task_new callable)
  if(NOT callable)
    return()
  endif()
  callable("${callable}")
  ans(callable)
  if(NOT callable)
    return()
  endif()

  map_new()
  ans(task)
  map_set_special(${task} $type task)
  task_queue_global()
  ans(default_task_queue)
  map_set("${task}" task_queue "${default_task_queue}")  
  map_set("${task}" callable "${callable}")
  map_set("${task}" arguments "${ARGN}")
  return(${task})
endfunction()

