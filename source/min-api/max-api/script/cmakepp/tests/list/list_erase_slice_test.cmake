function(test)


  set(lstA a b c d e f g)

  list_erase_slice(lstA 2 4)
  ans(res)


  assert(${lstA} EQUALS a b e f g)
  assert(${res} EQUALS c d)


endfunction()