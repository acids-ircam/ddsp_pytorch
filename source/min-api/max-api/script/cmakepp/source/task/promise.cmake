## `(<task>|<promise>|<any>)-><promise>` 
##
## transforms the input into a promise
## if the input is a promise it is directly retunred
## if input is  a task it is transformed into a promise
## if input is anything else it is wrapped inside a resolved promise
function(promise)
  set(promise "${ARGN}")
  is_promise("${promise}")
  ans(is_promise)
  if(NOT is_promise)
    
    is_task("${promise}")
    ans(is_task)
    if(is_task)
      promise_from_task("${promise}")
      ans(promise)
    else()
      promise_from_value("${promise}")
      ans(promise)
    endif()
  endif()
  task_queue_global()
  ans(task_queue)
  map_set_default("${promise}" task_queue "${task_queue}")  

  return_ref(promise)
endfunction()

function(promise_new)
  map_new()
  ans(promise)
  map_set_special("${promise}" $type promise)
  map_set("${promise}" promise_state "pending")
  return_ref(promise)
endfunction()

function(promise_from_value)
  promise_new()
  ans(promise)
  map_set("${promise}" promise_state "resolved")
  map_set("${promise}" value "${ARGN}")
  return_ref(promise)
endfunction()
  
function(promise_from_task)
  promise_new()
  ans(promise)

  task("${ARGN}")
  ans(task)

  if(NOT task)
    return()
  endif()

  map_set("${promise}" task "${task}")
  
  return(${promise})
endfunction()


function(promise_from_callable callable)
  task_new("${callable}")
  ans(task)
  promise_from_task("${task}")
  ans(promise)
  return_ref(promise)
endfunction()

function(promise_from_anonymous)
  arguments_anonymous_function(0 ${ARGC})
  ans(function)
  promise_from_callable("${function}")
  return_ans()
endfunction()



