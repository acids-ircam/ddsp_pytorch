function(test)


  set(lstA a b c)
  set(lstB a)
  set(lstC)



  list_replace_at(lstA 1 d)
  ans(res)
  assert(res)
  assert(${lstA} EQUALS a d c)


  list_replace_at(lstA 0 e)
  ans(res)
  assert(res)
  assert(${lstA} EQUALS e d c)


  list_replace_at(lstA 2 f)
  ans(res)
  assert(res)
  assert(${lstA} EQUALS e d f)


  list_replace_at(lstA 3 g)
  ans(res)
  assert(NOT res)
  assert(${lstA} EQUALS e d f)


  list_replace_at(lstC 0 a)
  ans(res)
  assert(NOT res)
  assert(${lstC} ISNULL)

endfunction()