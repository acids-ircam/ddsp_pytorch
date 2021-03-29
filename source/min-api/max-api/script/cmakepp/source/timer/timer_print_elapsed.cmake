## prints elapsed time for timer identified by id
function(timer_print_elapsed id)
  timer_elapsed("${id}")
  ans(elapsed)
  message("${ARGN}${id}: ${elapsed} ms")
  return()
endfunction()
