function(test)

  set(lstA a b c d)
  set(lstB a)
  set(lstC)


  list_peek_front(lstA)
  ans(res)
  assert("${res}" STREQUAL "a")


  list_peek_front(lstB)
  ans(res)
  assert("${res}" STREQUAL "a")


  list_peek_front(lstC)
  ans(res)
  assert(${res} ISNULL)
endfunction()