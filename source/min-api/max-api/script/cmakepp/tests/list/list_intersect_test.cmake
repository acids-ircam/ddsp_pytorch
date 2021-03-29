function(test)
    set(lstA 1 2 3 3)
  set(lstB 2 3 4)
  set(lstC 3 4 5)

  list_intersect()
  ans(res)
  assert(NOT res)

  list_intersect(lstA)
  ans(res)
  assert(EQUALS ${res} 1 2 3)

  list_intersect(lstA lstB)
  ans(res)
  assert(EQUALS ${res} 2 3)

  list_intersect(lstA lstB lstC)
  ans(res)
  assert(EQUALS ${res} 3)



  
endfunction()