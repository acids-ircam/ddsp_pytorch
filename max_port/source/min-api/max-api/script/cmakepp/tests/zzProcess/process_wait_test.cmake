function(test)

  ## arrange 

  # create process which runs long
  process_timeout(40)
  ans(handle)


  process_wait(${handle} --timeout 3)
  ans(failed_to_time_out)


  process_kill(${handle})

  process_wait(${handle})
  ans(successfully_stopped_before_timeout)


  assert(NOT failed_to_time_out AND successfully_stopped_before_timeout)

endfunction()