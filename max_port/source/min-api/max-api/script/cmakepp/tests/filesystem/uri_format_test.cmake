function(test)

  uri_to_localpath("\"c:\\test\\a b\\test.txt\"")
  ans(res)

  assert("${res}" STREQUAL "c:/test/a b/test.txt")



endfunction()



