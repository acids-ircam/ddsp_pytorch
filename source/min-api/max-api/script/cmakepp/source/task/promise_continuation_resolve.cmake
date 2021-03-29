## `(<promise>)-><void>`
##
## internal function which resolves a continuation
function(continuation_resolve promise continuation)
  promise_is_resolved("${continuation}")
  ans(continuation_is_resolved)
  if(continuation_is_resolved)
    return()
  endif()
  
  map_tryget("${continuation}" task)
  ans(continuation_task)


  map_tryget("${promise}" value)
  ans(value)
  ## decide depending on value and continuation_task 
  ## what to do.
  if(continuation_task)  
    # task_anonymous("" ()
    #   #message("resolving ${promise}'s task continuation ${continuation} with ${value}")
    #   map_set(${continuation_task} arguments ${value})
    #   set(promise ${continuation})
    #   task_invoke(${continuation_task})
    #   ans(result)
    #   #message("task returned \${result}")
    #   promise_resolve("${continuation}" "\${result}")
    #   )
    address_new()
    ans(value_adr)
    address_set("${value_adr}" "${value}")
    task_from_cmake_code("
      address_get(${value_adr})
      ans(value)
      map_set(${continuation_task} arguments \"\${value}\")
      set(promise ${continuation})
      task_invoke(${continuation_task})
      ans(result)
      promise_resolve(${continuation} \"\${result}\")
      ")
    ans(task)
  else()
    # task_anonymous("" ()
    #   #message("resolving ${promise}'s value continuation ${continuation} with ${value}")
    #   promise_resolve("${continuation}" "${value}")
    #   )
    # ans(task)
    task_from_cmake_code("promise_resolve(\"${continuation}\" \"${value}\")")
    ans(task)
  endif()


  ## use the task_queue provided by the continuation 
  ## if that fails use the task queue from the previous promise
  ## if that failes use the global task queue
  map_tryget("${continuation}" task_queue)
  ans(task_queue)
  if(NOT task_queue)
    map_tryget("${promise}" task_queue)
    ans(task_queue)
    if(NOT task_queue)
      message(FATAL_ERROR "no task queue specified")
    endif()
    map_set("${continuation}" task_queue "${task_queue}")
  endif()
  ## append task to task queue
  task_queue_push("${task_queue}" "${task}")


  return_ref(task)
endfunction()
