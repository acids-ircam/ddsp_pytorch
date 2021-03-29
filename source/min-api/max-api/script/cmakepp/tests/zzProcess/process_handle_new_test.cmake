function(test)


  timer_start(t1)
  foreach(i RANGE 0 10)
    process_start_info_new(COMMAND echo ads)
    ans(pi)
    process_handle_new(${pi})
  endforeach()
  timer_print_elapsed(t1)

endfunction()