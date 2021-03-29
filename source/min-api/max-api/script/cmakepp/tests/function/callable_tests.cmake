function(test)
  callable("[](a) return_math('{{a}} + {{a}}')")
  ans(uut)

  assert(uut)

  callable_call("${uut}" 3)
  ans(res)
  assert("${res}" EQUAL "6")

  is_callable("${uut}")
  ans(is_callable)
  assert(is_callable)

  ## works for cmake code
  callable("function()\nreturn(hello)\nendfunction()")
  ans(uut)
  assert(uut)

  callable_call(${uut})
  ans(res)
  assert("${res}" STREQUAL "hello")


  ## works for lambdas
  call2("[]()return(1)")
  ans(res)
  assert("${res}" STREQUAL 1)

  ## works for cmake code
  call2("function(__)\nreturn(3)\nendfunction()")
  ans(res)
  assert("${res}" STREQUAL 3)


  function(f2)
    return(4)
  endfunction()
  ## works for cmake functions
  call2(f2)
  ans(res)
  assert(${res} STREQUAL 4)

  ## works for cmake files
  fwrite("asd.cmake" "function(fk)\n return(9)\nendfunction()")
  call2("asd.cmake")
  ans(res)
  assert(${res} STREQUAL 9)


  ## performacne comparison
  ## eval is the fastest dynamic call call2 is slower (1/4 slower)

  function(f1 a b)
    return_math("${a} + ${b}")
  endfunction()


  set(len 1000)

   timer_start(original_call)
  foreach(i RANGE 0 ${len})
    call(f1(1 2))
  endforeach()
  timer_print_elapsed(original_call)


  timer_start(call2)
  foreach(i RANGE 0 ${len})
    call2(f1 1 2)
  endforeach()
  timer_print_elapsed(call2)


  timer_start(callable_call)
  foreach(i RANGE 0 ${len})

    callable("f1")
    callable_call("${__ans}" 1 2)
  endforeach()
  timer_print_elapsed(callable_call)


  timer_start(callable_function_eval)
  foreach(i RANGE 0 ${len})
    callable_function("f1")
    eval("${__ans}(1 2)")
  endforeach()
  timer_print_elapsed(callable_function_eval)

  timer_start(pure_eval)
  foreach(i RANGE 0 ${len})
    eval("f1(1 2)")
  endforeach()

  timer_print_elapsed(pure_eval)


endfunction()