function(test)

  string_encode_bracket("asb[asddasdw]asdwad")
  ans(res)
  assert("${res}" MATCHES "asb.asddasdw.asdwad")

  string_decode_bracket("${res}")
  ans(res)
  assert("${res}" "asb[asddasdw]asdwad" EQUALS)

endfunction()