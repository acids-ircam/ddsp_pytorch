function(test)


  ## tests weather process_kill terminates a running process

  ## arrange
  process_timeout(100)
  ans(handle)


  process_isrunning(${handle})
  ans(success)

  ## act
  process_kill(${handle})
  ans(stopping)


  process_isrunning(${handle})
  ans(failed)

  ## assert
  assert(success AND stopping AND NOT failed)

endfunction()