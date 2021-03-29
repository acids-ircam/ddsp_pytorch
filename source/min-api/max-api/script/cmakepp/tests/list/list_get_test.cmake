function(test)

  set(lstC a b c)

  list_get(lstC -2)
  ans(res)
  assert(${res} STREQUAL "c")

  list_get(lstC 0)
  ans(res)
  assert(${res} STREQUAL "a")



endfunction()