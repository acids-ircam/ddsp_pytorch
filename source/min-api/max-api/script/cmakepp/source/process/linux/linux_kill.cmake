

  ## wraps the linux pkill command
  function(linux_kill)
    wrap_executable(linux_kill kill)
    linux_kill(${ARGN})
    return_ans()
  endfunction()
