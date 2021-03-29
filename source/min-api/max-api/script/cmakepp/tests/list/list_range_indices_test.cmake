function(test)



  set(lst a b c d)
  list_range_indices(lst 1 2 3)
  ans(res)
  assert(${res} EQUALS 1 2 3)


  list_range_indices(lst :)
  ans(res)
  assert(${res} EQUALS 0 1 2 3 )


endfunction()