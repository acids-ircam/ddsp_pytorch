function(test)


  set(lstA a b c d)
  set(lstB a d b c)
  set(lstC a d b c k)

  set_isequal(lstA lstB)
  ans(res)
  assert(res)

  set_isequal(lstA lstC)
  ans(res)
  assert(NOT res)

endfunction()