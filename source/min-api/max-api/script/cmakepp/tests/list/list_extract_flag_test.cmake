function(test)

  set(lstA a b c e d e f g)

  list_extract_flag(lstA e)
  ans(res)
  assert(res)
  assert(${lstA} EQUALS a b c d e f g)



  list_extract_flag(lstA k)
  ans(res)
  assert(NOT res)

endfunction()