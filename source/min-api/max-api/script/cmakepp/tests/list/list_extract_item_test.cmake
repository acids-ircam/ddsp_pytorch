function(test)




  set(lst 1 2 3)
  list_extract_flag(lst 2)
  ans(res)
  assert(res)
  assert(EQUALS ${lst} 1 3)

  list_extract_flag(lst 2)
  ans(res)
  assert(NOT res)
  assert(EQUALS ${lst} 1 3)

  list_extract_flag(lst 1)
  ans(res)
  assert(res)
  assert(EQUALS ${lst} 3)

  list_extract_flag(lst 1)
  ans(res)
  assert(NOT res)
  assert(EQUALS ${lst} 3)

  list_extract_flag(lst 3)
  ans(res)
  assert(res)
  assert(NOT lst)

  set(lst)
  list_extract_flag(lst 1)
  ans(res)
  assert(NOT res)




endfunction()