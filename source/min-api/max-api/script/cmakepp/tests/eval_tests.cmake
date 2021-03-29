function(test)

  set(a)
  set(code "set(a b)")
  eval("${code}")
  assert(NOT a)


  eval_ref(code)
 # assert("${a}" STREQUAL b)


  set(n 1000)



  set(code "")
  timer_start(eval_ref)
  foreach(i RANGE 0 ${n})
    eval_ref(code)
  endforeach()
  timer_print_elapsed(eval_ref)


  timer_start(eval)
  foreach(i RANGE 0 ${n})
    eval("")
  endforeach()
  timer_print_elapsed(eval)

endfunction()