function(test)
  



  set(script "
    foreach(i RANGE 0 10)
      message(\${i})
      execute_process(COMMAND \${CMAKE_COMMAND} -E sleep 1)
    endforeach()
    message(end)
    ")

  process_start_script("${script}")
  ans(pi1)
  message(started1)
  process_start_script("${script}")
  ans(pi2)
  message(started2)
  process_start_script("${script}")
  ans(pi3)
  message(started3)



  process_wait_all(${pi1} ${pi2} ${pi3})
  ans(res)

  process_wait_all(${pi1} ${pi2} ${pi3})
  ans(res)

  endfunction()