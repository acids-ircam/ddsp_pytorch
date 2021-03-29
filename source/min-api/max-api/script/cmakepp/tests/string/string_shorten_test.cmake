function(test)
  set(res "")
  # Shorten to one char (+ shortener)
  set(input "a long string")
  string_shorten("${input}" 4)
  ans(res)
  assert("${res}" STREQUAL "a...")

  set(res "")
  # Empty string (1/2)
  set(input "")
  string_shorten("${input}" 1)
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Empty string (2/2)
  set(input "")
  string_shorten("${input}" 1)
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Shortener "." + 3 chars from input
  set(input "a long string")
  string_shorten("${input}" 4 ".")
  ans(res)
  assert("${res}" STREQUAL "a l.")

  set(res "")
  # Only shortener (default: 3 chars long) is returned
  set(input "a long string")
  string_shorten("${input}" 3)
  ans(res)
  assert("${res}" STREQUAL "...")

  set(res "")
  # Shortener too long for max_length, empty string returned
  set(input "a long string")
  string_shorten("${input}" 2)
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Shortener + first char (1/2)
  set(input "a long string")
  string_shorten("${input}" 2 ".")
  ans(res)
  assert("${res}" STREQUAL "a.")

  set(res "")
  # Shortener + first char (2/2)
  set(input "a long string")
  string_shorten("${input}" 3 "..")
  ans(res)
  assert("${res}" STREQUAL "a..")

  set(res "")
  # Max_length > string length: full string returned
  set(input "word")
  string_shorten("${input}" 5)
  ans(res)
  assert("${res}" STREQUAL "word")

  set(res "")
  # Max_length == string length: full string returned
  set(input "word")
  string_shorten("${input}" 4)
  ans(res)
  assert("${res}" STREQUAL "word")

  set(res "")
  # Max_length == 0: empty string returned
  set(input "word")
  string_shorten("${input}" 0)
  ans(res)
  assert("${res}_" STREQUAL "_")
endfunction()