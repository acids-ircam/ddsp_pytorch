function(test)

  index_range(0 10)
  ans(range)

  set(empty_lst)

  list_contains(range 3)
  ans(res)
  assert(res)


  list_contains(range 11)
  ans(res)
  assert(NOT res)


  list_contains(empty_lst 2)
  ans(res)
  assert(NOT res)



endfunction()