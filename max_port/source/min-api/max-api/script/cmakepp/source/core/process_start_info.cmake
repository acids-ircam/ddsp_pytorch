## takes a ~<command line> or ~<process start info>
## and returns a valid  process start info
function(process_start_info)
  set(__args ${ARGN})

  list_extract_labelled_value(__args TIMEOUT)
  ans(timeout_arg)

  list_extract_labelled_value(__args WORKING_DIRECTORY)
  ans(cwd_arg)

  if("${ARGN}_" STREQUAL "_")
    return()
  endif()


  obj("${ARGN}")
  ans(obj)

  if(NOT obj)
    command_line(${__args})
    ans(obj)
  endif()


  if(NOT obj)
    message(FATAL_ERROR "invalid process start info ${ARGN}")
  endif()

  set(path)
  set(cwd)
  set(command)
  set(args)
  set(parameters)
  set(timeout)
  set(arg_string)
  set(command_string)

  scope_import_map(${obj})

  if("${args}_" STREQUAL "_")
    set(args ${parameters})
  endif()

  if("${command}_" STREQUAL "_")
    set(command "${path}")
    if("${command}_" STREQUAL "_")
      message(FATAL_ERROR "invalid <process start info> missing command property")
    endif()
  endif()

  if(timeout_arg)
    set(timeout "${timeout_arg}")
  endif()

  if("${timeout}_" STREQUAL "_" )
    set(timeout -1)
  endif()




  if(cwd_arg)
    set(cwd "${cwd_arg}")
  endif()

  path("${cwd}")
  ans(cwd)

  if(EXISTS "${cwd}")
    if(NOT IS_DIRECTORY "${cwd}")
      message(FATAL_ERROR "specified working directory path is a file not a directory: '${cwd}'")
    endif()
  else()
    message(FATAL_ERROR "specified workind directory path does not exist : '${cwd}'")
  endif()



  # create a map from the normalized input vars
  map_capture_new(command args cwd timeout)
  return_ans()

endfunction()