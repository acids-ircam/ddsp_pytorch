function(test)
  set(res "")
  # Remove last char
  set(input "abc")
  string_remove_ending("${input}" "c")
  ans(res)
  assert("${res}" STREQUAL "ab")

  set(res "")
  # Remove last two chars
  set(input "abc")
  string_remove_ending("${input}" "bc")
  ans(res)
  assert("${res}" STREQUAL "a")

  set(res "")
  # Remove all chars
  set(input "abc")
  string_remove_ending("${input}" "abc")
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Length of ending greater than input (1/2)
  set(input "abc")
  string_remove_ending("${input}" "abcd")
  ans(res)
  assert("${res}" STREQUAL "abc")

  set(res "")
  # Length of ending greater than input (2/2)
  set(input "abc")
  string_remove_ending("${input}" "abcde")
  ans(res)
  assert("${res}" STREQUAL "abc")

  set(res "")
  # Empty input string
  set(input "")
  string_remove_ending("${input}" "a")
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Empty ending string
  set(input "abc")
  string_remove_ending("${input}" "")
  ans(res)
  assert("${res}" STREQUAL "abc")

  set(res "")
  # Empty input and ending string
  set(input "")
  string_remove_ending("${input}" "")
  ans(res)
  assert("${res}_" STREQUAL "_")
endfunction()