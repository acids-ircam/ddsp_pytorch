function(test)

  set(lstA a b c d e)
  set(lstB c d e)
  set(lstC a b k)


  set_issubset(lstB lstA)
  ans(res)
  assert(res)

  set_issubset(lstC lstA)
  ans(res)
  assert(NOT res)
endfunction()