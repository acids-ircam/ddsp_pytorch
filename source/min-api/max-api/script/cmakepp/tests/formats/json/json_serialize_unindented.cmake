function(test)

  
  json("123123")
  ans(res)
  assert("${res}" STREQUAL "123123")

  json("absdasd")
  ans(res)
  assert("${res}" STREQUAL "\"absdasd\"")


  json("ab\\sdasd")
  ans(res)
  assert("${res}" STREQUAL "\"ab\\\\sdasd\"")


  map()
    kv(a 1)
  end()
  ans(map)
  json("${map}")
  ans(res)
  assert("${res}" STREQUAL "{\"a\":1}")


  json(1 2 3 4)
  ans(res)
  assert("${res}" STREQUAL "[1,2,3,4]")


  map_new()
  ans(res)
  json(${res})
  ans(res)
  assert("${res}" STREQUAL "{}")

endfunction()