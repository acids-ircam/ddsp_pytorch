function(test)
  
  
  


  range_from_indices(1 2 3)
  ans(res)
  assert(${res} EQUALS [1:3])

  range_from_indices()
  ans(res)
  assert(${res} ISNULL)

  range_from_indices(3 2 1)
  ans(res)
  assert(${res} EQUALS [3:1:-1])


  range_from_indices(1)
  ans(res)
  assert(${res} EQUALS 1)

  range_from_indices(1 2)
  ans(res)
  assert(${res} EQUALS "1 2")

  range_from_indices(1 2 3 4 5 8)
  ans(res)
  assert(${res} EQUALS "[1:5] 8")


  range_from_indices(8 5 2 8 3 4 7 10 1 2 3 5 3)
  ans(res)
  assert(${res} EQUALS "[8:2:-3] 8 3 [4:10:3] [1:3] 5 3")

endfunction()