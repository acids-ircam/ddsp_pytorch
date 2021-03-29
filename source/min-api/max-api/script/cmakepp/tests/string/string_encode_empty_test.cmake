function(test)

  string_encode_empty("")
  ans(res)

  assert(res)

  string_decode_empty("${res}")
  ans(res)
  assert(NOT res)


  string_decode_empty("asdasd")
  ans(res)

  assert("${res}" STREQUAL "asdasd")


endfunction()