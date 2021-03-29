function(test)
  set(len 20)

  timer_start(t1)
  foreach(i RANGE 0 ${len})
    execute_process(COMMAND ${CMAKE_COMMAND} -E echo_append a)
  endforeach()
  timer_print_elapsed(t1)

  timer_start(t1)
  foreach(i RANGE 0 ${len})
    execute(COMMAND ${CMAKE_COMMAND} -E echo_append a)
    
  endforeach()
  timer_print_elapsed(t1)


endfunction()