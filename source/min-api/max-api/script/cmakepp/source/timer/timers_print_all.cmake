## prints the elapsed time for all known timer
function(timers_print_all)
  timers_get()
  ans(timers)
  foreach(timer ${timers})
    timer_print_elapsed("${timer}")
  endforeach()  
  return()
endfunction()