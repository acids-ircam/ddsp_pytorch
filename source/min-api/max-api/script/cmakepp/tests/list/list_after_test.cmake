function(test)



  set(lst a b c --asd d e f)
  list_after(lst --asd)
  ans(res)
  assert(${res} EQUALS d e f)

  set(lst a b c d e f)
  list_after(lst --asd)
  ans(res)
  assert(NOT res)

  set(lst --asd a b c d e f)
  list_after(lst --asd)
  ans(res)
  assert(${res} EQUALS a b c d e f )

  set(lst)
  list_after(lst --asd)
  ans(res)
  assert(NOT res)
endfunction()