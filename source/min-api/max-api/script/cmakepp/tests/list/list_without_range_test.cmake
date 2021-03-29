function(test)


  set(lstA a b c d e f)

  list_without_range(lstA 2 4)
  ans(res)

  assert(${res} EQUALS a b e f)

endfunction()