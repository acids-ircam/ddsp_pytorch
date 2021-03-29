function(test)
 
  set(lst 0 1)
  list_combinations(lst lst lst)
  ans(res)
  assert(${res} CONTAINS 000)
  assert(${res} CONTAINS 001)
  assert(${res} CONTAINS 010)
  assert(${res} CONTAINS 011)
  assert(${res} CONTAINS 100)
  assert(${res} CONTAINS 101)
  assert(${res} CONTAINS 110)
  assert(${res} CONTAINS 111)


  set(listA a b)
  set(listB 1 2)
  set(listC I II)
  set(listD one)

  list_combinations()
  ans(res)
  assert(NOT res)


  list_combinations(listD)
  ans(res)
  assert(EQUALS ${res} one)

  list_combinations(listA)
  ans(res)
  assert(EQUALS ${res} a b)

  list_combinations(listA listB)
  ans(res)
  assert(EQUALS ${res} a1 a2 b1 b2)

  list_combinations(listA listB listC)
  ans(res)
  assert(EQUALS ${res} a1I a1II a2I a2II b1I b1II b2I b2II)





endfunction()