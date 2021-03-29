

  ## platform specific implementaiton for process_kill
  function(process_kill_Linux handle)
    process_handle("${handle}")
    ans(handle)

    map_tryget(${handle} pid)
    ans(pid)

    linux_kill(-SIGTERM ${pid} --exit-code)
    ans(error)

    return_truth("${error}" EQUAL 0)
  endfunction() 