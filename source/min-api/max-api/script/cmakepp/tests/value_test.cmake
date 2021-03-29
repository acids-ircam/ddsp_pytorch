function(test)

  value()
  ans(res)
  assert("${res}_" STREQUAL _)


  value("hello")
  ans(res)
  assert("${res}" STREQUAL "hello")

  value(a b c)
  ans(res)
  assert(${res} EQUALS a b c)

  value("[]() return(a b c {{ARGN}})" d e)
  ans(res)
  assert(${res} EQUALS a b c d e)

  function(abc)
    set(res "c" "d" "e" ${ARGN})
    return_ref(res)
  endfunction()
  


  value(abc d e)
  ans(res)
  assert(${res} EQUALS c d e d e)

  value("{a:1}" b "{b:2}")
  ans(res)
  assertf({res[0].a} EQUAL 1)
  assertf({res[1]} STREQUAL "b")
  assertf({res[2].b} EQUAL 2)


  

endfunction()