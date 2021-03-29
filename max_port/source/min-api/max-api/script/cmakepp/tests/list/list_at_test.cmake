function(test)




  set(lstA a b c d e f g)


  list_at(lstA 0 0 2 2 -2 -3 -4)
  ans(res)
  assert(${res} EQUALS a a c c g f e)



endfunction()