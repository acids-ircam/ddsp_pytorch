function(test)

  set(init_list 0 1 2 3 4 5 6 7 8 9 a b c d e f)


  set(lst ${init_list})
  list_range_remove(lst "1:4 6:10 12:$")
  ans(res)
  assert(${res} EQUALS 13)
  assert(${lst} EQUALS 0 5 b)

  set(lst ${init_list})
  list_range_remove(lst ":")
  ans(res)
  assert(${res} EQUALS 16)
  assert(${lst} ISNULL)

  set(lst ${init_list})
  list_range_remove(lst $)
  ans(res)
  assert(${res} EQUALS 1)
  assert(${lst} EQUALS 0 1 2 3 4 5 6 7 8 9 a b c d e)


  set(lst ${init_list})
  list_range_remove(lst "")
  ans(res)
  assert(${res} EQUALS 0)
  assert(${lst} EQUALS 0 1 2 3 4 5 6 7 8 9 a b c d e f)


endfunction()