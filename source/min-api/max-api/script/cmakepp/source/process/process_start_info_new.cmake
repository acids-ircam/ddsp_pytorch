## `(<command string>|<object> [TIMEOUT <n:int>] [WORKING_DIRECTORY ~<path>] [--passthru])-><process start info>` 
## `<command string> ::= "COMMAND"? <command> <arg...>` 
##
## creates a new process_start_info with the following fields
## ```
## <process start info> ::= {
##   command: <executable> 
##   command_arguments: <encoded list>
##   working_directory: <directory>
##   timeout: <n>
##   passthru: true|false
## }
## ```
## the syntax of the process_start_info_new is equivalent to cmake's built in `execute_process` command.
##
##
## the command needs to point to an executable,  preferrably a fully qualified path 
## the command_arguments contain an encoded list.  you can specify any argument in the execute function - they will be passed along as you write them  (this deserves extra appreciation because doing so in cmake is hard)
## If the flags you want to pass along to the process conflict with the flags of the process function you can always encode them yourself and pass them along as an encoded list
##
##
## *example*
##  * `process_start_info_new(COMMAND cmake -E echo "asd bsd" csd) -> <% process_start_info_new(COMMAND cmake -E echo "asd;bsd")
##  ans(info)
##  template_out_json(${info}) %>` 
function(process_start_info_new)
  arguments_encoded_list(0 ${ARGC})
  ans(arguments_list)

  list_extract_any_labelled_value(arguments_list WORKING_DIRECTORY)
  ans(working_directory)
  list_extract_any_labelled_value(arguments_list TIMEOUT)
  ans(timeout)

  list_extract_flag(arguments_list --passthru)
  ans(passthru)

  if(NOT timeout)
    set(timeout -1)
  endif()

  path_qualify(working_directory)

  list_pop_front(arguments_list)
  ans(command)

  if("${command}_" STREQUAL "COMMAND_")
    list_pop_front(arguments_list)
    ans(command)
  endif()

  string_decode_list("${command}")
  ans(command)

  map_new()
  ans(process_start_info)
  map_set(${process_start_info} command "${command}")  
  map_set(${process_start_info} command_arguments "${arguments_list}")
  map_set(${process_start_info} working_directory "${working_directory}")  
  map_set(${process_start_info} timeout "${timeout}")
  map_set(${process_start_info} passthru "${passthru}")
  return_ref(process_start_info)
endfunction()
