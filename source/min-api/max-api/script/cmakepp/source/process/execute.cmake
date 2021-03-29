## `(<process start info> [--process-handle] [--exit-code] [--async] [--silent-fail] [--success-callback <callable>]  [--error-callback <callable>] [--state-changed-callback <callable>])-><process handle>|<exit code>|<stdout>|<null>`
##
## *options*
## * `--process-handle` 
## * `--exit-code` 
## * `--async` 
## * `--silent-fail` 
## * `--success-callback <callable>[exit_code](<process handle>)` 
## * `--error-callback <callable>[exit_code](<process handle>)` 
## * `--state-changed-callback <callable>[old_state;new_state](<process handle>)` 
## * `--lean`
## *example*
## ```
##  execute(cmake -E echo_append hello) -> '@execute(""cmake", "-E", "echo_append", "hello"")'
## ```
function(execute)
  set(args ${ARGN})
  list_extract_flag(args --lean)
  ans(lean)


  arguments_encoded_list(0 ${ARGC})
  ans(args)

  list_extract_flag(args --process-handle)
  ans(return_handle)
  list_extract_flag(args --exit-code)
  ans(return_exit_code)
  list_extract_flag(args --async)
  ans(async)
  #list_extract_flag(args --async-wait)
  #ans(wait)
  #if(wait)
  #  set(async true)
  #endif()
  list_extract_flag(args --silent-fail)
  ans(silent_fail)

  list_extract_labelled_value(args --success-callback)
  ans(success_callback)
  list_extract_labelled_value(args --error-callback)
  ans(error_callback)
  list_extract_labelled_value(args --state-changed-callback)
  ans(process_callback)
  list_extract_labelled_value(args --on-terminated-callback)
  ans(terminated_callback)
  if(NOT args)
    messagE(FATAL_ERROR "no command specified")
  endif()

  process_start_info_new(${args})
  ans(start_info)

##debug here
  #print_vars(start_info.command start_info.command_arguments)

  process_handle_new(${start_info})
  ans(process_handle)

  if(success_callback)
    string_decode_list("${success_callback}")
    ans(success_callback)
    assign(success = process_handle.on_success.add("${success_callback}"))
  endif()
  if(error_callback)
    string_decode_list("${error_callback}")
    ans(error_callback)
    assign(success = process_handle.on_error.add("${error_callback}"))
  endif()
  if(process_callback)
    string_decode_list("${process_callback}")
    ans(process_callback)
    assign(success = process_handle.on_state_change.add("${process_callback}"))
  endif()
  if(terminated_callback)
    string_decode_list("${terminated_callback}")
    ans(terminated_callback)
    assign(success = process_handle.on_terminated.add("${terminated_callback}"))
  endif()

  if(async)
    process_start(${process_handle})
    return(${process_handle})
  else()
    process_execute(${process_handle})
    if(return_handle)
      return(${process_handle})
    endif()


    
    map_tryget(${process_handle} exit_code)
    ans(exit_code)

    if(return_exit_code)
      return_ref(exit_code)
    endif()

    map_tryget(${process_handle} pid)
    ans(pid)
    if(NOT pid)
      message(FATAL_ERROR FORMAT "could not find command '{start_info.command}'")
    endif()

    if(exit_code AND silent_fail)
      error("process {start_info.command} failed with {process_handle.exit_code}")
      return()
    endif()

    if(exit_code)
      message(FATAL_ERROR FORMAT "process {start_info.command} failed with {process_handle.exit_code}")
    endif()


    map_tryget(${process_handle} stdout)
    ans(stdout)
    return_ref(stdout)

  endif()


endfunction()