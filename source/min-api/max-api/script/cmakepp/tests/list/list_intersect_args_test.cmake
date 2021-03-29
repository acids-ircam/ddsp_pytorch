function(test)

  set(flags --a --b --c --d)
  list_intersect_args(flags --c --d --e)
  ans(res)
  assert(EQUALS ${res} --c --d)


  set(flags)
  list_intersect_args(flags --c --e)
  ans(res)
  assert(NOT res)


  list_intersect_args(flags)
  ans(res)
  assert(NOT res)

  set(flags --a --b --c)
  list_intersect_args(flags)
  ans(res)
  assert(NOT res)

endfunction()