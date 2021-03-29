function(test)
  set(res "")
  # Empty string
  string_normalize("")
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # No change (1/2)
  string_normalize("abc")
  ans(res)
  assert("${res}" STREQUAL "abc")

  set(res "")
  # No change (2/2)
  string_normalize("abcABC1234567890")
  ans(res)
  assert("${res}" STREQUAL "abcABC1234567890")

  set(res "")
  # Whitespace to underscore
  string_normalize("a bc")
  ans(res)
  assert("${res}" STREQUAL "a_bc")

  set(res "")
  # Whitespace and punctuation to underscore
  string_normalize("a bc!?.")
  ans(res)
  assert("${res}" STREQUAL "a_bc___")
endfunction()