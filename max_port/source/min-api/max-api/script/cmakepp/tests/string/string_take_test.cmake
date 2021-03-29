function(test)
  set(res "")
  # The char "w" at beginning is removed
  set(input "word")
  string_take(input "w")
  ans(res)
  assert("${res}" STREQUAL "w")
  assert("${input}" STREQUAL "ord")

  set(res "")
  # The chars "wo" at beginning are removed
  set(input "word")
  string_take(input "wo")
  ans(res)
  assert("${res}" STREQUAL "wo")
  assert("${input}" STREQUAL "rd")

  set(res "")
  # Whitespaces at the beginning are not removed
  set(input "  whitespaces")
  string_take(input "  ")
  ans(res)
  assert("${res}" STREQUAL "  ")
  assert("${input}" STREQUAL "whitespaces")

  set(res "")
  # Whitespaces in the middle are kept
  set(input "white  spaces")
  string_take(input "white")
  ans(res)
  assert("${res}" STREQUAL "white")
  assert("${input}" STREQUAL "  spaces")

  set(res "")
  # Empty match string
  set(input "word")
  string_take(input "")
  ans(res)
  assert("${res}_" STREQUAL "_")
  assert("${input}" STREQUAL "word")
  
  set(res "")
  # Empty input string
  set(input "")
  string_take(input "a")
  ans(res)
  assert("${res}_" STREQUAL "_")
  assert("${input}_" STREQUAL "_")

  set(res "")
  # Empty input/match string
  set(input "")
  string_take(input "")
  ans(res)
  assert("${res}_" STREQUAL "_")
  assert("${input}_" STREQUAL "_")
endfunction()