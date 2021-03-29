function(test)



  set(lstA a b c)
  list_to_string(lstA " ")
  ans(res)
  assert("${res}" STREQUAL "a b c")
endfunction()