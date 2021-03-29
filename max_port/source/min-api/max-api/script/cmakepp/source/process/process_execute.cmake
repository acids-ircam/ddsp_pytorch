## `(<process handle>)-><process handle>`
##
## executes the specified command with the specified arguments in the 
## current working directory
## creates and registers a process handle which is then returned 
## this function accepts arguments as encoded lists. this allows you to include
## arguments which contain semicolon and other special string chars. 
## the process id of processes start with `process_execute` is always -1
## because `CMake`'s `execute_process` does not return it. This is not too much of a problem
## because the process will always be terminated as soon as the function returns
## 
## **parameters**
##   * `<command>` the executable (may contain spaces) 
##   * `<arg...>` the arguments - may be an encoded list 
## **scope**
##   * `pwd()` used as the working-directory
## **events** 
##   * `on_process_handle_created` global event is emitted when the process_handle is ready
##   * `process_handle.on_state_changed`
##
## **returns**
## ```
## <process handle> ::= {
##   pid: "-1"|"0"
##     
## }
## ``` 
function(process_execute process_handle)
  process_handle_register(${process_handle})

  map_tryget(${process_handle} start_info)
  ans(process_start_info)  

  ## the pid is -1 by default for non async processes
  map_set(${process_handle} pid -1)

  ## register process handle
  process_handle_change_state(${process_handle} starting)
  process_handle_change_state(${process_handle} running)

  map_tryget(${process_start_info} working_directory)
  ans(cwd)



  map_tryget(${process_start_info} command)
  ans(command)

  cmake_string_escape("${command}")
  ans(command)

  map_tryget(${process_start_info} command_arguments)
  ans(command_arguments)


  map_tryget(${process_start_info} passthru)
  ans(passthru)


  #command_line_args_combine(${command_arguments})
  #ans(command_arguments_string)

  set(command_arguments_string)
  foreach(argument ${command_arguments})
    string_decode_list("${argument}")
    ans(argument)
    cmake_string_escape("${argument}")
    ans(argument)
    set(command_arguments_string "${command_arguments_string} ${argument}")
  endforeach()
  

  map_tryget(${process_start_info} timeout)
  ans(timeout)


  if("${timeout}" GREATER -1)
    set(timeout TIMEOUT ${timeout})
  else()
    set(timeout)
  endif()

  #message("executing ${command} ${command_arguments_string}")


  set(output_handling "  OUTPUT_VARIABLE stdout\n  ERROR_VARIABLE stderr")

  if(passthru)
    set(output_handling)
  endif()


  set(eval_this "
    execute_process(
      COMMAND ${command} ${command_arguments_string}
      ${output_handling}
      RESULT_VARIABLE exit_code
      WORKING_DIRECTORY ${cwd}
      ${timeout}
    )
  ")
#  _message("${eval_this}")
  eval_ref(eval_this)

  ## set process handle variables
  if(NOT "${exit_code}" MATCHES "^-?[0-9]+$")
    map_set(${process_handle} pid)
  endif()
  map_set(${process_handle} exit_code "${exit_code}")
  map_set(${process_handle} stdout "${stdout}")
  map_set(${process_handle} stderr "${stderr}")

  ## change state
  process_handle_change_state(${process_handle} terminated)

  return_ref(process_handle)
endfunction()
