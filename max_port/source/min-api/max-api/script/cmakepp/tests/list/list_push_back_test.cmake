function(test)

  set(lstA)
  set(lstB a)
  set(lstC a b c)



  list_push_back(lstA d)
  assert(${lstA} EQUALS d)

  list_push_back(lstB d)
  assert(${lstB} EQUALS a d)


  list_push_back(lstC d)
  assert(${lstC} EQUALS a b c d)



endfunction()