function(test)

  set(lstA 1 2 3 3)
  set(lstB 2 3 4)
  set(lstC 3 4 5)

  list_union()
  ans(res)
  assert(NOT res)

  list_union(lstA)
  ans(res)
  assert(EQUALS ${res} 1 2 3)

  list_union(lstA lstB)
  ans(res)
  assert(EQUALS ${res} 1 2 3 4)

  list_union(lstA lstB lstC)
  ans(res)
  assert(EQUALS ${res} 1 2 3 4 5)


endfunction()