function(test)

  cmake_token_range("a b c d")
  ans_extract(begin end)

  cmake_token_range_serialize("${begin};${end}")
  ans(res)

  cmake_token_range_insert(${end} " e f g")
  cmake_token_range_insert(${begin} "1 2 3 ")
  ans_extract(begin)
  cmake_token_range_serialize("${begin};${end}")
  ans(res)

  assert("${res}" STREQUAL "1 2 3 a b c d e f g")


endfunction()