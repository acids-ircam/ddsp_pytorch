function(test)

  

  command_line_args_combine(a b c)
  ans(res)
  assert("${res}" STREQUAL "a b c")

  command_line_args_combine("a b" c)
  ans(res)
  assert("${res}" STREQUAL "\"a b\" c")

  command_line_args_combine(a b c)
  ans(res)
  assert("${res}" STREQUAL "a b c")

  command_line_args_combine(a "\"b" c )
  ans(res)
  assert("${res}" STREQUAL "a \"\\\"b\" c")

endfunction()