function(test)

  set(lstA a b c d e f g)
  set(lstB)

  list_find(lstA d)
  ans(res)
  assert("${res}" EQUAL 3)


  list_find(lstA a)
  ans(res)
  assert("${res}" EQUAL 0)

  list_find(lstA h)
  ans(res)
  assert("${res}" EQUAL -1)

  list_find(lstB h)
  ans(res)
  assert("${res}" EQUAL -1)

endfunction()