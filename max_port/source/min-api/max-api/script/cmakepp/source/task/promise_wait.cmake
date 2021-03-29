## `(<promise> [--ticks <uint>] [--timeout <uint>])-><any>...`
##
## waits for the specified promise to complete. causes the execution of 
## the promises task queue.
## --ticks and --timout indicate constaints on how long the tasks will run
function(promise_wait promise)
  map_tryget("${promise}" task_queue)
  ans(task_queue)
  task_queue_run("${task_queue}" ${ARGN})
  map_tryget("${promise}" value)
  return_ans()
endfunction()
