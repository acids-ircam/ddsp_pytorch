function(test)

  anonymous_function()
  ans(res)
  assert(NOT res)

  anonymous_function(() return(hi))
  ans(res)
  assert(res)
  is_anonymous_function("${res}")
  ans(it_is)
  assert(it_is)

  anonymous_function("${res}")
  ans(res2)
  assert(res2)

  assert("${res2}" STREQUAL "${res}")

endfunction()