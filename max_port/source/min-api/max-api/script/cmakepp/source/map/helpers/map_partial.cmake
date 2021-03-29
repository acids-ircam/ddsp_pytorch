function(test)
  new()
  ans(obj)
  obj_set(${obj} "test1" "val1")
  obj_set(${obj} "test2" "val2")
  obj_set(${obj} "test3" "val3")


  obj_pick("${obj}" test1 test3)
  ans(res)
  assert(DEREF {res.test1} STREQUAL "val1")
  assert(DEREF {res.test3} STREQUAL "val3")

  obj_pick("${obj}" test4)
  ans(res)
  assert(res)
  assert(DEREF "_{res.test4}" STREQUAL "_")


endfunction()