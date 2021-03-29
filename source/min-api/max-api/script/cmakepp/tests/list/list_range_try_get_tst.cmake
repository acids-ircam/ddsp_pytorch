function(test)




  set(lst a b c d e)
  list_range_try_get(lst 1 2 3 9 10)
  ans(res)
  assert(${res} EQUALS b c d)


endfunction()