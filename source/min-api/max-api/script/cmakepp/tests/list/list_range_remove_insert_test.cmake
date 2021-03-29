function(test)



  set(lst a b c d e f g)


  set(lstA)
  list_range_partial_write(lstA "[]" a)
  assert(${lstA} EQUALS a)


  set(lstA ${lst})
  list_range_partial_write(lstA "1:3" 1 2 3)
  assert(${lstA} EQUALS a 1 2 3 e f g) 



  set(lstA ${lst})
  list_range_partial_write(lstA "1:3")
  assert(${lstA} EQUALS a e f g)


  set(lstA ${lst})
  list_range_partial_write(lstA "" 9)
  assert(${lstA} EQUALS a b c d e f g 9)


  set(lstA)
  list_range_partial_write(lstA "")
  ans(res)
  assert(${lstA} ISNULL)


  set(lstA ${lst})
  list_range_partial_write(lstA "[2)" 1 2 3)
  assert(${lstA} EQUALS a b 1 2 3 c d e f g)




endfunction()