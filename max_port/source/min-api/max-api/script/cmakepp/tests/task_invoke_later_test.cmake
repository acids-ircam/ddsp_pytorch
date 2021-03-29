function(test)

  task_enqueue("[]()message('hello')")
  task_enqueue("[]()message('hello1')")
  task_enqueue("[]()message('hello2')")
  task_enqueue("[]()message('hello3')")
  tqr()

endfunction()