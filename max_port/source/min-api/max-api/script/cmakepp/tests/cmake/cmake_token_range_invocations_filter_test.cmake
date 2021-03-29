function(test)

  cmake_invocation_filter_token_range("
    asd(a)
    bsd(a)
    bsd(b)
    csd(c)
    " invocation_identifier STREQUAL bsd --reverse --take 1)
  ans(res)
  assertf({res[0].invocation_identifier} STREQUAL "bsd")
  assertf({res[0].invocation_arguments} STREQUAL "b")


  cmake_invocation_filter_token_range("
    asd(a)
    bsd(a)
    bsd(b)
    csd(c)
    " invocation_identifier STREQUAL bsd AND invocation_arguments STREQUAL a --reverse --take 1)
  ans(res)
  assertf({res[0].invocation_identifier} STREQUAL "bsd")
  assertf({res[0].invocation_arguments} STREQUAL "a")
  return()

  cmake_invocation_filter_token_range("function(ddd)\nasd(dasd)\nendfunction()\nset(a b c d e f g)" true --skip 1 --take 2)
  ans(res)
  assertf("{res[0].invocation_identifier}" STREQUAL asd)
  assertf("{res[0].invocation_token.value}" STREQUAL "asd" )
  assertf("{res[1].invocation_identifier}" STREQUAL endfunction)
  assertf("{res[0].invocation_arguments}" EQUALS dasd)
  assertf("{res[1].invocation_arguments}" ISNULL)


  cmake_invocation_filter_token_range("function(ddd)\nasd(dasd)\nendfunction()\nset(a b c d e f g)" {invocation_identifier} STREQUAL "asd")
  ans(res)
  assertf("{res[0].invocation_identifier}" STREQUAL asd)

endfunction()