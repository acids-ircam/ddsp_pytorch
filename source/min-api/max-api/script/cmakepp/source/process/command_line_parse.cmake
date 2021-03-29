## command_line_parse 
## parses the sepcified cmake style  command list which starts with COMMAND 
## or parses a single command line call
## returns a command line object:
## {
##   command:<string>,
##   args: <string>...
## }
function(command_line_parse)
  set(args ${ARGN})

  if(NOT args)
    return()
  endif()


  list_pop_front(args)
  ans(first)

  list(LENGTH args arg_count)

  if("${first}_" STREQUAL "COMMAND_")
    list_pop_front(args)
    ans(command)

    command_line_args_combine(${args})
    ans(arg_string)


    set(command_line "\"${command}\" ${arg_string}")      
  else()
    if(arg_count)
     message(FATAL_ERROR "either use a single command string or a list of 'COMMAND <command> <arg1> <arg2> ...'")
    endif()
    set(command_line "${first}")
  endif()


  command_line_parse_string("${command_line}")
  return_ans()
endfunction()