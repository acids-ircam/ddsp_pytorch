function(command_line_to_string)
    command_line(${ARGN})
    ans(cmd)

    scope_import_map(${cmd})

    command_line_args_combine(${args})
    ans(arg_string)
    if(NOT "${arg_string}_" STREQUAL "_")
      set(arg_string " ${arg_string}")
    endif()
    set(command_line "${command}${arg_string}")
    return_ref(command_line)
  endfunction()


  