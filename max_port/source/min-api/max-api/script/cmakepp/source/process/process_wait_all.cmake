## `(<handles: <process handle...>>  [--timeout <seconds>] [--idle-callback <callable>] [--task-complete-callback <callable>] )`
##
## waits for all specified <process handles> to finish returns them in the order
## in which they completed
##
## `--timeout <n>`    if value is specified the function will return all 
##                    finished processes after n seconds
##
## `--idle-callback <callable>`   
##                    if value is specified it will be called at least once
##                    and between every query if a task is still running 
##
##
## `--task-complete-callback <callable>`
##                    if value is specified it will be called whenever a 
##                    task completes.
##
## *Example*
## `process_wait_all(${handle1} ${handle1}  --task-complete-callback "[](handle)message(FORMAT '{handle.pid}')")`
## prints the process id to the console whenver a process finishes
##
function(process_wait_all)
  set(args ${ARGN})

  list_extract_labelled_value(args --idle-callback)
  ans(idle_callback)

  list_extract_labelled_value(args --task-complete-callback)
  ans(task_complete_callback)

  list_extract_labelled_value(args --timeout)
  ans(timeout)
  set(timeout_task_handle)


  process_handles(${args})
  ans(process_list)
  ans(running_processes)


  list(LENGTH running_processes process_count)

  set(timeout_process_handle)
  if(timeout)
    process_timeout(${timeout})
    ans(timeout_process_handle)
    list(APPEND running_processes ${timeout_process_handle})
  endif()
  set(complete_count 0)
  while(running_processes)

    list_pop_front(running_processes)
    ans(current_process)
    process_refresh_handle(${current_process})
    ans(is_running)
    
    #message(FORMAT "{current_process.pid} is_running {is_running} {current_process.state} ")

    if(NOT is_running)
      if("${current_process}_" STREQUAL "_${timeout_process_handle}")
        set(running_processes)
      else()          

        list(APPEND complete_processes ${current_process})          
        if(NOT quietly)
          list(LENGTH complete_processes complete_count)           
          if(task_complete_callback)
            call2("${task_complete_callback}" "${current_process}") 
          endif()
        endif() 
      endif()        
    else()
      ## insert into back
      list(APPEND running_processes ${current_process})
    endif()

    if(idle_callback)
      call2("${idle_callback}")
    endif()

  endwhile()

  return_ref(complete_processes)
endfunction()
