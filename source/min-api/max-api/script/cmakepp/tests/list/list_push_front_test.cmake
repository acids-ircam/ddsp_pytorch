function(test)

  set(lstA a b c)
  set(lstB a)
  set(lstC)


  list_push_front(lstA d)
  ans(res)
  assert(res)
  assert(${lstA} EQUALS d a b c)




endfunction()