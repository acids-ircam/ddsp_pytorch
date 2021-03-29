function(test)

  set(lstA a b c d e f g)

  list_remove_at(lstA 6 1 3 )
  assert(${lstA} EQUALS a c e f)


endfunction()