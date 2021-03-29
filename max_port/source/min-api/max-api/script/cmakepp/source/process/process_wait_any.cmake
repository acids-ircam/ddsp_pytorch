  ## waits until any of the specified handles stops running
  ## returns the handle of that process
  ## if --timeout <n> is specified function will return nothing after n seconds
  function(process_wait_any)
    set(args ${ARGN})

    list_extract_flag(args --quietly)
    ans(quietly)    

    process_handles(${args})
    ans(processes)


    if(NOT quietly)
      list(LENGTH processes len)
      echo_append("waiting for any of ${len} processes to finish.")  
    endif()

    set(timeout_process_handle)
    if(timeout)
      process_timeout(${timeout})
      ans(timeout_process_handle)
      list(APPEND processes ${timeout_process_handle})
    endif()

    while(processes)
      list_pop_front(processes)
      ans(process)
      process_refresh_handle(${process})
      ans(isrunning)


      if(NOT quietly)
        tick()
      endif()

      if(NOT isrunning)
        if("${process}_" STREQUAL "${timeout_process_handle}_")      
          echo(".. timeout")
          return()
        endif()
        if(NOT quietly)
          echo("")
        endif()
        return(${process})
      else()
        list(APPEND processes ${process})
      endif()

    endwhile()   
  endfunction()