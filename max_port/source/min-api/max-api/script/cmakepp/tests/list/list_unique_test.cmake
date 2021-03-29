function(test)

  set(lstA a b c d)
  set(lstB a a b a c a b c c d d)
  set(lstC a)
  set(lstD)

  list_unique(lstA)
  ans(res)
  assert(${res} EQUALS a b c d)

  list_unique(lstB)
  ans(res)
  assert(${res} EQUALS a b c d)

  list_unique(lstC)
  ans(res)
  assert(${res} EQUALS a)

  list_unique(lstD)
  ans(res)
  assert(${res} ISNULL)


endfunction()