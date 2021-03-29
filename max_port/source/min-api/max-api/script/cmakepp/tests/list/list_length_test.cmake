function(test)

  set(lstA a b c)
  list_length(lstA)
  ans(res)

  assert("${res}" EQUAL 3)

  set(lstB)
  list_length(lstB)
  ans(res)
  assert("${res}" EQUAL 0)

endfunction()