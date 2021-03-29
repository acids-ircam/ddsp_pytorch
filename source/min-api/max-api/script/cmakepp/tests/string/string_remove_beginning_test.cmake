function(test)
  set(res "")
  # First first char
  set(input "abc")
  string_remove_beginning("${input}" "a")
  ans(res)
  assert("${res}" STREQUAL "bc")

  set(res "")
  # Remove first two chars
  set(input "abc")
  string_remove_beginning("${input}" "ab")
  ans(res)
  assert("${res}" STREQUAL "c")

  set(res "")
  # Remove all chars
  set(input "abc")
  string_remove_beginning("${input}" "abc")
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Length of ending greater than input
  set(input "abc")
  string_remove_beginning("${input}" "abcd")
  ans(res)
  assert("${res}" STREQUAL "abc")

  set(res "")
  # Empty input string
  set(input "")
  string_remove_beginning("${input}" "a")
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Empty ending string
  set(input "abc")
  string_remove_beginning("${input}" "")
  ans(res)
  assert("${res}" STREQUAL "abc")

  set(res "")
  # Empty input and ending string
  set(input "")
  string_remove_beginning("${input}" "")
  ans(res)
  assert("${res}_" STREQUAL "_")
endfunction()