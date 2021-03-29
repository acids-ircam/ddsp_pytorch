function(test)

  set(lstA a b c d e)

  list_erase(lstA 1 3)
  ans(res)
  assert(${res} EQUALS ${lstA})
  assert(${lstA} EQUALS a d e )

  

endfunction()