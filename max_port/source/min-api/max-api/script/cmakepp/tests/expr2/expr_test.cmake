function(test)

  function(my_echo_func inp)
    return("${inp}")
  endfunction()

  function(my_echo_func_2_args inp inpTwo)
    return("${inpTwo} ${inp}")
  endfunction()

  expr(a)
  ans(res)
  assert("${res}" STREQUAL a)

  expr(my_echo_func("Some echo"))
  ans(res)
  assert("${res}" STREQUAL "Some echo")

  expr(my_echo_func_2_args("Some", "echo"))
  ans(res)
  assert("${res}" STREQUAL "echo Some")

endfunction()