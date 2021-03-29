function(test)





  set(lstA)
  set(lstB a)
  set(lstC a b c)

  list_contains_any(lstA)
  ans(res)
  assert(res)

  list_contains_any(lstA a)
  ans(res)
  assert(NOT res)

  list_contains_any(lstB)
  ans(res)
  assert(res)

  list_contains_any(lstB b)
  ans(res)
  assert(NOT res)

  list_contains_any(lstB a)
  ans(res)
  assert(res)

  list_contains_any(lstC)
  ans(res)
  assert(res)

  list_contains_any(lstC d)
  ans(res)
  assert(NOT res)

  list_contains_any(lstC d e)
  ans(res)
  assert(NOT res)

  list_contains_any(lstC a)
  ans(res)
  assert(res)

  list_contains_any(lstC a c )
  ans(res)
  assert(res)

  list_contains_any(lstC b d e)
  ans(res)
  assert(res)

endfunction()