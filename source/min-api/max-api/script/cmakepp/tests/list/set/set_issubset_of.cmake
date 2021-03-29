function(test)

  set(nullSet)
  set(set1 2)
  set(setA 2 3)
  set(setB 1 2 3)

  set_issubset(nullSet nullSet)
  ans(res)
  assert(res)

  set_issubset(nullSet set1)
  ans(res)
  assert(res)


  set_issubset(set1 nullSet)
  ans(res)
  assert(NOT res)

  set_issubset(setA setB)
  ans(res)
  assert(res)


  set_issubset(setB setA)
  ans(res)
  assert(NOT res)




endfunction()