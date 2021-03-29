function(test)

  command_line_args_escape("a")
  ans(res)
  assert("${res}" STREQUAL "a")

  command_line_args_escape("a b")
  ans(res)
  assert("${res}" STREQUAL "\"a b\"")

  command_line_args_escape("\"")
  ans(res)
  assert("${res}" STREQUAL "\"\\\"\"")

  command_line_args_escape("dada\"asdasd")
  ans(res)
  assert("${res}" STREQUAL "\"dada\\\"asdasd\"")
  



  command_line_args_escape(a b c "hallo du" "asdfdef" "lalala\"\"")
  ans(res)

  assert(${res} a b c "\"hallo du\"" asdfdef "\"lalala\\\"\\\"\"" ARE_EQUAL)



endfunction()