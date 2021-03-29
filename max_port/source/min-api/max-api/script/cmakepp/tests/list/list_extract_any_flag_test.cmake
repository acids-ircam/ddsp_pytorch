function(test)

  set(lstA a b c d e f)

  list_extract_any_flag(lstA c d f)
  ans(res)
  assert(res)
  assert(${lstA} EQUALS a b e)

  
  
endfunction()