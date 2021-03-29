## `(<task>)-><true>` 
##
## invokes the specified task
## scope: 
## * `task` instance of the task currently being invoked
## * `task_queue` if invoked in a task_queue
## * `arguments` contains the arguments for the specified task
function(task_invoke task)
  map_tryget(${task} callable)
  ans(callable)
  map_tryget(${task} arguments)
  ans(arguments)
  call2("${callable}" ${arguments})
  ans(res)

  map_set(${task} return_value ${res})
  return_ref(res)    
endfunction()




