


  ## creates a default handler from the specified cmake function
  function(handler_default func)
    if(NOT COMMAND "${func}")
      return()
    endif()
      function_new()
      ans(call_function)
      function_import("
        function(funcname request response)
          map_tryget(\${request} input)
          ans(input)
          ${func}(\"\${input}\")
          ans(res)
          map_set(\${response} output \"\${res}\")
          return(true)
        endfunction()
        " as ${call_function} REDEFINE)

    data("{
      callable:$call_function,
      display_name:$func,
      labels:$func
      }")
    ans(handler)

    request_handler("${handler}")
    return_ans()

  endfunction()

