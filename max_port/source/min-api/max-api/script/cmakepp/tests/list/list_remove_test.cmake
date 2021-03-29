function(test)
  set(lstA a b c d e )
  set(lstB)


  list_remove(lstA c d a)
  assert(${lstA} EQUALS b e)

  list_remove(lstB a b c)
  assert(${lstB} ISNULL)

endfunction()