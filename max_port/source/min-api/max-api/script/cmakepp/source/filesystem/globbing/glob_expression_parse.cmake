
  ## glob_expression_parse(<glob ignore path...>) -> {include:<glob path>, exclude:<glob path>}
  ##
  ##
  function(glob_expression_parse)
    set(args ${ARGN})

    is_map("${args}")
    ans(ismap)
    if(ismap)
      return_ref(args)
    endif()

    string(REGEX MATCHALL "![^;]+" exclude "${args}")
    string(REGEX MATCHALL "[^!;]+" exclude "${exclude}")
    string(REGEX MATCHALL "(^|;)[^!;][^;]*" include "${args}")
    string(REGEX MATCHALL "[^;]+" include "${include}")


    map_capture_new(include exclude)
    ans(res)
    return_ref(res)

  endfunction()
