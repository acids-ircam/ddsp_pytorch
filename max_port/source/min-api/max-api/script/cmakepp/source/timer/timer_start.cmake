## starts a timer identified by id
## 
function(timer_start id)
  map_set_hidden(__timers __prejudice 0)

  # actual implementation of timer_start
  function(timer_start id)
    return_reset()      
    millis()
    ans(millis)
    map_set(__timers ${id} ${millis})
  endfunction()



  ## this is run the first time a timer is started: 
  ## it calculates a prejudice value 
  ## (the time it takes from timer_start to timer_elapsed to run)
  ## this prejudice value is then subtracted everytime elapse is run
  ## thus minimizing the error

  #foreach(i RANGE 0 3)
    timer_start(initial)  
    timer_elapsed(initial)
    ans(prejudice)

    map_tryget(__timers __prejudice)
    ans(pre)
    math(EXPR prejudice "${prejudice} + ${pre}")
    map_set_hidden(__timers __prejudice ${prejudice})
  #endforeach()


  timer_delete(initial)


  return_reset()
  timer_start("${id}")
endfunction()