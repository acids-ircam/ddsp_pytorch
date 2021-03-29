function(test)


  set(str "aa bb cc \"d e\" f \"ll\\\"oo\"")
  string_take_commandline_arg(str)
  ans(res)
  assert("${res}" STREQUAL "aa")
  string_take_commandline_arg(str)
  ans(res)
  assert("${res}" STREQUAL "bb")
  string_take_commandline_arg(str)
  ans(res)
  assert("${res}" STREQUAL "cc")
  string_take_commandline_arg(str)
  ans(res)
  assert("${res}" STREQUAL "d e")

  string_take_commandline_arg(str)
  ans(res)
  assert("${res}" STREQUAL "f")

  string_take_commandline_arg(str)
  ans(res)
  assert("${res}" STREQUAL "ll\"oo")



endfunction()