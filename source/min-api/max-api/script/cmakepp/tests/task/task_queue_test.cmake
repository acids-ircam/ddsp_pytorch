function(test)



  task_queue_new()
  ans(uut)

  task_anonymous("1;2" (a b) return("\${a} \${b} \${a} \${b}"))
  ans(task)
  task_queue_push("${uut}" ${task})
  task_queue_tick("${uut}")
  ans(run_task)
  assert("${task}" STREQUAL "${run_task}")
  assertf("{task.return_value}" STREQUAL "1 2 1 2")


  ## neverending task queue stopped by timeout
  task_queue_new()
  ans(uut)
  task_anonymous("" () task_queue_push("${uut}" "\${task}"))
  ans(task)
  timer_start(t1)
  ## timeout after 1000 ms
  task_queue_push("${uut}" ${task})
  task_queue_run("${uut}" --timeout 1000)
  ans(ticks)
  timer_print_elapsed(t1)
  print_vars(ticks)

  ## neverending task queue stopped by ticks
  task_queue_new()
  ans(uut)
  task_anonymous("" () task_queue_push("${uut}" \${task}))
  ans(task)
  task_queue_push("${uut}" "${task}")
  timer_start(t1)
  task_queue_run("${uut}" --ticks ${ticks})
  timer_print_elapsed(t1)

  timer_start(t1)
  task_queue_run("${uut}" --ticks 1)
  timer_elapsed(t1)
  ans(t_overhead)

  timer_start(task_run_500_ticks)
  task_queue_run("${uut}" --ticks 500)
  ans(ticks)
  timer_print_elapsed(task_run_500_ticks)
  timer_elapsed(task_run_500_ticks)
  ans(time)
  math(EXPR tpt "${time} / ${ticks}")
  message("a single task took ${tpt} ms to run (overhead is ${t_overhead} ms)")



endfunction()