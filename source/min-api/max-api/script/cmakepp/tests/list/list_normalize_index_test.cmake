function(test)


  set(lstA)

  list_normalize_index(lstA 0)
  ans(res)
  assert("${res}" EQUAL 0)

  list_normalize_index(lstA -1)
  ans(res)
  assert("${res}" EQUAL 0)


  set(lstB a b c)
  list_normalize_index(lstB -2)
  ans(res)
  assert(${res} EQUAL 2)

endfunction()