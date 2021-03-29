function(test)

  set(lstB)
  list_range_replace(lstB "")
  ans(res)
  assert(${res} ISNULL)
  assert(${lstB} ISNULL)

  set(lstB a b c)
  list_range_replace(lstB "")
  ans(res)
  assert(${res} ISNULL)
  assert(${lstB} EQUALS a b c)

  #       0 1 2 3
  set(lstB a b c )
  list_range_replace(lstB "*")
  ans(res)
  assert(${res} EQUALS a b c)
  assert(${lstB} ISNULL)

  #       0 1 2 3 4 5
  set(lstB a b c d e )
  list_range_replace(lstB "1:3")
  ans(res)
  assert(${res} EQUALS b c d)
  assert(${lstB} EQUALS a e)


  set(lst a b c d e)
  list_range_replace(lst "4 0 3:1:-2" 1 2 3 4)
  ans(res)
  assert(${res} EQUALS e a d b)
  assert(${lst} EQUALS 2 4 c 3 1)

endfunction()