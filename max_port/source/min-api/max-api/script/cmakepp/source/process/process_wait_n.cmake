
## `(<n:<int>|"*"> <process handle>... [--idle-callback:<callable>])-><process handle>...`
##
## waits for at least <n> processes to complete 
##
## returns: 
##  * at least `n` terminated processes
## 
## arguments: 
## * `n` an integer the number of processes to return (lower bound) if `n` is clamped to the number of processes. if `n` is * it is replaced with the number of processes 
## * `--idle-callback` is called after every time a processes state was polled. It is guaranteed to be called once per process handle. it has access to the following scope variables
##    * `terminated_count` number of terminated processes
##    * `running_count` number of running processes
##    * `wait_time` time that was waited
##    * `wait_counter` number of times the waiting loop iterated
##    * `running_processes` list of running processes 
##    * `current_process` the current process being polled
##    * `is_running` the running state of the current process
##    * `terminated_processes` the list of terminated processes
##
function(process_wait_n n)
  arguments_extract_typed_values(0 ${ARGC}
    <n:<string>>
    [--idle-callback:<callable>] # 
    [--timeout:<int>?]
  )
  ans(process_handles)

  list(LENGTH process_handles process_count)

  set(running_processes ${process_handles})
  set(terminated_processes)

  timer_start(__process_wait_timer)
  set(wait_counter 0)
  
  if("${n}" STREQUAL "*")
    list(LENGTH running_processes n)
  endif()  

  set(terminated_count 0)

  set(wait_time)
  while(true)
    if(timeout)
      if(${timeout} GREATER ${wait_time})
        break()
      endif()
    endif()


    set(queue ${running_processes})

    while(queue)
      list_pop_front(queue)
      ans(current_process)
    

      process_refresh_handle(${current_process})
      ans(is_running)

      if(NOT is_running)
        list(REMOVE_ITEM running_processes ${current_process})
        list(APPEND terminated_processes ${current_process})
      endif()
      
      ## status vars
      timer_elapsed(__process_wait_timer)
      ans(wait_time)
      list(LENGTH terminated_processes terminated_count)
      list(LENGTH running_processes running_count)

      if(idle_callback)
        call2("${idle_callback}")
      endif()
      math(EXPR wait_counter "${wait_counter} + 1")

    endwhile()

    if(NOT ${terminated_count} LESS ${n})
      break()
    endif()
    if(NOT running_processes)
      return()
    endif()
  endwhile()


  return_ref(terminated_processes)
endfunction()