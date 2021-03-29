function(test)
  
  set(lstA a b c d)

  list_get_labelled_value(lstA b)
  ans(res)
  assert("${res}" STREQUAL "c")


endfunction()