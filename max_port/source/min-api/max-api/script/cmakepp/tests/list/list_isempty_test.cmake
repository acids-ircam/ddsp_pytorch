function(test)


  set(lstA)
  set(lstB a)
  set(lstC a b)

  list_isempty(lstA)
  ans(res)
  assert(res)

  list_isempty(lstB)
  ans(res)
  assert(NOT res)


  list_isempty(lstC)
  ans(res)
  assert(NOT res)

endfunction()