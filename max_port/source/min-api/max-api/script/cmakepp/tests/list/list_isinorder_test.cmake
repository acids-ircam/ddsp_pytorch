function(test)

  set(lstA a b c d e)

  list_isinorder(lstA a b)
  ans(res)
  assert(res)

  list_isinorder(lstA b d )
  ans(res)
  assert(res)

  list_isinorder(lstA e d)
  ans(res)
  assert(NOT res)

endfunction()