## `(<process start info>)-><process handle>`
##
## returns a new process handle which has the following layout:
## ```
## <process handle> ::= {
##   pid: <pid>  
##   start_info: <process start info>
##   state: "unknown"|"running"|"terminated"
##   stdout: <text>
##   stderr: <text>
##   exit_code: <integer>|<error string>
##   command: <executable>
##   command_args: <encoded list>
##   on_state_change: <event>[old_state, new_state](${process_handle}) 
## }
## ``` 
function(process_handle_new start_info)
  map_new()
  ans(process_handle)
  map_set(${process_handle} pid "")
  map_set(${process_handle} start_info "${start_info}")
  map_set(${process_handle} state "unknown")
  map_set(${process_handle} stdout "")
  map_set(${process_handle} stderr "")
  map_set(${process_handle} exit_code)
  event_new()
  ans(event)
  map_set(${process_handle} on_state_change ${event})

  event_new()
  ans(event)
  map_set(${process_handle} on_success ${event})


  event_new()
  ans(event)
  map_set(${process_handle} on_error ${event})


  event_new()
  ans(event)
  map_set(${process_handle} on_terminated ${event})

  return_ref(process_handle)
endfunction()
