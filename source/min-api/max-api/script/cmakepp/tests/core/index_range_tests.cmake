function(test)
  # index_range tests
  index_range(0 0)
  ans(res)
  assert(NOT res)

  index_range(1 1)
  ans(res)
  assert(NOT res)


  index_range(-1 -1)
  ans(res)
  assert(NOT res)

  index_range(0 1)
  ans(res)
  assert(EQUALS ${res} 0)

  index_range(0 2)
  ans(res)
  assert(EQUALS ${res} 0 1)

  index_range(2 0)
  ans(res)
  assert(EQUALS ${res} 2 1)

  index_range(4 8)
  ans(res)
  assert(EQUALS ${res} 4 5 6 7)

  index_range(8 4)
  ans(res)
  assert(EQUALS ${res} 8 7 6 5)
endfunction()