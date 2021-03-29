function(test)

  set(lstA a b c d e f g)
  list_set_at(lstA 3 k)
  assert(${lstA} EQUALS a b c k e f g)
endfunction()