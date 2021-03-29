function(test)

  set(lstB)
  set(lstA 1 2 3 --test1 a --test2 b --test3 [a b c] )


  list_extract_labelled_value(lstB any)
  ans(res)
  assert(NOT res)
  assert(NOT lstB)

  list_extract_labelled_value(lstA --test1)
  ans(res)
  assert("${res}" STREQUAL a)
  assert(EQUALS ${lstA} 1 2 3 --test2 b --test3 [a b c])


  list_extract_labelled_value(lstA --test4)
  ans(res)
  assert(NOT res)  
  assert(EQUALS ${lstA} 1 2 3 --test2 b --test3 [a b c])


  list_extract_labelled_value(lstA --test3)
  ans(res)
  assert(EQUALS ${res} a b c)
  assert(EQUALS ${lstA} 1 2 3 --test2 b)


  list_extract_labelled_value(lstA --test2)
  ans(res)
  assert(EQUALS ${res} b)
  assert(EQUALS ${lstA} 1 2 3)

  set(lstA --test1)
  list_extract_labelled_value(lstA --test1)
  ans(res)
  assert(NOT res)
  assert(NOT lstA)
  


endfunction()