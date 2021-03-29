function(test)
  
  string(ASCII 31 separator)

  set(uut ";")
  string_encode_semicolon("${uut}")
  ans(res)
  assert("${res}" STREQUAL "${separator}")
  

  string_decode_semicolon("${separator}")
  ans(res)
  assert("_${res}_" EQUALS "_;_")

  

endfunction()