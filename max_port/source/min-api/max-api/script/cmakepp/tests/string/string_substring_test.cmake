function(test)
  set(res "")
  # First char is discarded
  set(input "word")
  string_substring("${input}" 1)
  ans(res)
  assert("${res}" STREQUAL "ord")

  set(res "")
  # First two chars are extracted
  set(input "word")
  string_substring("${input}" 0 2)
  ans(res)
  assert("${res}" STREQUAL "wo")

  set(res "")
  # Second and third chars are extracted
  set(input "word")
  string_substring("${input}" 1 2)
  ans(res)
  assert("${res}" STREQUAL "or")

  set(res "")
  # Empty string
  set(input "")
  string_substring("${input}" 0)
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Negative index wrap around
  set(input "word")
  string_substring("${input}" -2)
  ans(res)
  assert("${res}" STREQUAL "d")

  set(res "")
  # Negative index wrap around with optional length param
  set(input "word")
  string_substring("${input}" -3 2)
  ans(res)
  assert("${res}" STREQUAL "rd")
endfunction()