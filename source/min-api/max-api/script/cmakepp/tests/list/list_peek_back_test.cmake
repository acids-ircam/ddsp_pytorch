function(test)

  set(lstA a b c)
  set(lstB a)
  set(lstC)


  list_peek_back(lstA)
  ans(res)
  assert("${res}" STREQUAL "c")


  list_peek_back(lstB)
  ans(res)
  assert("${res}" STREQUAL "a")


  list_peek_back(lstC)
  ans(res)
  assert("${res}" ISNULL)


endfunction()