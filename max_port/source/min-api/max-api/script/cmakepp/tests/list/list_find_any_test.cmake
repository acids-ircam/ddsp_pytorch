function(test)

  set(lstA)
  set(lstB a b c d)

  list_find_any(lstA a b c d) 
  ans(res)
  assert("${res}" EQUAL -1)



  list_find_any(lstB d a)
  ans(res)
  assert(NOT "${res}" EQUAL -1)
  list_get(lstB ${res})
  ans(itm)
  assert("${itm}" STREQUAL "d" OR "${itm}" STREQUAL "a")

endfunction()