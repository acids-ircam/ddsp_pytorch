function(test)
  
  string_encode_delimited("hello ' there!" '')
  ans(res)
  assert("${res}" STREQUAL "'hello \\' there!'")

  string_decode_delimited("${res}" '')
  ans(res)
  assert("${res}" STREQUAL "hello ' there!")



  string_encode_delimited("hello \" there!" \")
  ans(res)
  assert("${res}" STREQUAL "\"hello \\\" there!\"")

  string_decode_delimited("${res}" \"\")
  ans(res)
  assert("${res}" STREQUAL "hello \" there!")



  string_encode_delimited("hello <> there!" "<>")
  ans(res)
  assert("${res}" STREQUAL "<hello <\\> there!>")

  string_decode_delimited("${res}" "<>")
  ans(res)
  assert("${res}" STREQUAL "hello <> there!")



endfunction()