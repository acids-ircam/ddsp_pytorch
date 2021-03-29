function(test)





  set(nullSet)
  set(oneSet a)
  set(setA a b c)
  set(setB a b c)
  set(setC a b )


  set_isequal(nullSet nullSet)
  ans(res)
  assert(res)

  set_isequal(oneSet nullSet)
  ans(res)
  assert(NOT res)

  set_isequal(oneSet oneSet)
  ans(res)
  assert(res)

  set_isequal(setA setB)
  ans(res)
  assert(res)

  set_isequal(setB setC)
  ans(res)
  assert(NOT res)








endfunction()