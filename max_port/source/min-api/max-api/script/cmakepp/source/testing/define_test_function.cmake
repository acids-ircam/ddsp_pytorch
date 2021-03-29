

  ## deprecated
  function(define_test_function name parse_function_name)
    set(args ${ARGN})
    list(LENGTH args arg_len)
    matH(EXPR arg_len "${arg_len} + 1")


    string_combine(" " ${args})
    ans(argstring)
    set(evaluated_arg_string)
    foreach(arg ${ARGN})
      set(evaluated_arg_string "${evaluated_arg_string} \"\${${arg}}\"")
    endforeach()
   # messagE("argstring ${argstring}")
   # message("evaluated_arg_string ${evaluated_arg_string}")
    eval("
      function(${name} expected ${argstring})
        arguments_encoded_list(${arg_len} \${ARGC})
        ans(encoded_arguments)
        arguments_sequence(${arg_len} \${ARGC})
        ans(arguments_sequence)
        set(args \${ARGN})
        list_extract_flag(args --print)
        ans(print)
        data(\"\${expected}\")
        ans(expected)
        #if(parsed)
        #  set(expected \${parsed})
        #endif()
        #if(NOT expected)
        #  message(FATAL_ERROR \"invalid expected value\")
        #endif()
        ${parse_function_name}(${evaluated_arg_string} \${args})
        ans(uut)

        if(print)
          json_print(\${uut})
        endif()


        
        map_match(\"\${uut}\" \"\${expected}\")
        ans(res)
        if(NOT res)
          echo_append(\"actual: \")
          json_print(\${uut})
          echo_append(\"expected: \")
          json_print(\${expected})
        endif()
        assert(res MESSAGE \"values do not match\")
      endfunction()

    ")
    return()
  endfunction()
