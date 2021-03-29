function(test)

  set(lstA a b c)

  list_swap(lstA 0 0)
  assert(${lstA} EQUALS a b c)

  list_swap(lstA 0 2)
  assert(${lstA} EQUALS c b a)

  

endfunction()