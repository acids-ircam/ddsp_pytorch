function(test)


  task_queue_global()
  ans(tq1)
  task_queue_global()
  ans(tq2)
  assert(tq1)
  assert("${tq1}" STREQUAL "${tq2}")

endfunction()