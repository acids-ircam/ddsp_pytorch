function(test)


  string_combine("," a b c d)
  ans(res)
  assert("${res}" STREQUAL "a,b,c,d")


  string_combine("," a)
  ans(res)
  assert("${res}" STREQUAL "a")


  string_combine(",")
  ans(res)
  assert("${res}_" STREQUAL "_")


  string_combine("")
  ans(res)
  assert("${res}_" STREQUAL "_")


  string_combine("" a b)
  ans(res)
  assert("${res}" STREQUAL "ab")
  

endfunction()