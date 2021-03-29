## refreshes the fields of the process handle
## returns true if the process is still running false otherwise
## this is the only function which is allowed to change the state of a process handle
function(process_refresh_handle handle)
  process_handle("${handle}")
  ans(handle)


  process_isrunning("${handle}")
  ans(isrunning)

  if(isrunning)
    set(state running)
  else()
    set(state terminated)
  endif()

  if("${state}" STREQUAL "terminated")
    process_return_code("${handle}")
    ans(exit_code)
    process_stdout("${handle}")
    ans(stdout)
    process_stderr("${handle}")
    ans(stderr)
    map_capture("${handle}" exit_code stdout stderr)
  endif()


  process_handle_change_state("${handle}" "${state}")
  ans(state_changed)

  




  return_ref(isrunning)

endfunction()
