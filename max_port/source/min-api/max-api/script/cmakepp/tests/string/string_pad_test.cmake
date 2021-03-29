function(test)
  set(res "")
  # Empty input, 2 whitespaces added
  set(input "")
  string_pad("${input}" 2)
  ans(res)
  assert("${res}" STREQUAL "  ")

  set(res "")
  # Empty input, no padding
  set(input "")
  string_pad("${input}" 0)
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # No padding (1/3)
  set(input "word")
  string_pad("${input}" 0)
  ans(res)
  assert("${res}" STREQUAL "word")

  set(res "")
  # No padding (2/3)
  set(input "word")
  string_pad("${input}" 3)
  ans(res)
  assert("${res}" STREQUAL "word")

  set(res "")
  # No padding (3/3)
  set(input "word")
  string_pad("${input}" 4)
  ans(res)
  assert("${res}" STREQUAL "word")

  set(res "")
  # 2 whitespaces added at the end
  set(input "word")
  string_pad("${input}" 6)
  ans(res)
  assert("${res}" STREQUAL "word  ")

  set(res "")
  # 2 "-" added at the end
  set(input "word")
  string_pad("${input}" 6 "-")
  ans(res)
  assert("${res}" STREQUAL "word--")

  ######################
  # Test for "--prepend"
  ######################
  set(flag "--prepend")

  set(res "")
  # Empty input, 2 whitespaces added
  set(input "")
  string_pad("${input}" 2 "${flag}")
  ans(res)
  assert("${res}" STREQUAL "  ")

  set(res "")
  # Empty input, no padding
  set(input "")
  string_pad("${input}" 0 "${flag}")
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # No padding (1/3)
  set(input "word")
  string_pad("${input}" 0 "${flag}")
  ans(res)
  assert("${res}" STREQUAL "word")

  set(res "")
  # No padding (2/3)
  set(input "word")
  string_pad("${input}" 3 "${flag}")
  ans(res)
  assert("${res}" STREQUAL "word")

  set(res "")
  # No padding (3/3)
  set(input "word")
  string_pad("${input}" 4 "${flag}")
  ans(res)
  assert("${res}" STREQUAL "word")

  set(res "")
  # 2 whitespaces added at the beginning
  set(input "word")
  string_pad("${input}" 6 "${flag}")
  ans(res)
  assert("${res}" STREQUAL "  word")

  set(res "")
  # 2 "-" added at the beginning
  set(input "word")
  string_pad("${input}" 6 "${flag}" "-")
  ans(res)
  assert("${res}" STREQUAL "--word")
endfunction()