function(test)

  function(uut)
    arguments_encoded_list(0 ${ARGC})
    ans(args)

    process_start_info_new(${args})
    ans(startinfo)

    process_handle_new(${startinfo})
    ans(handle)
    
    process_start(${handle})
    return_ans()
  endfunction()

  uut(COMMAND "ping.exe" "-t" "heise.de")
  ans(handle)

  process_kill("${handle}")

  process_wait(${handle})




endfunction()