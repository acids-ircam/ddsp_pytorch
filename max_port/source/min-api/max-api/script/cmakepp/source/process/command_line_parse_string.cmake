

  function(command_line_parse_string str)
    uri_parse("${str}")
    ans(uri)

    map_tryget(${uri} rest)
    ans(rest)   


    uri_to_localpath("${uri}")
    ans(command)
    
    set(args)
    while(true)
      string_take_commandline_arg(rest)
      ans(arg)
      string_decode_delimited("${arg}")
      ans(arg)

      list(APPEND args "${arg}")
      if("${arg}_" STREQUAL "_")
        break()
      endif()
    endwhile()

    map_capture_new(command args)
    return_ans()
  endfunction()