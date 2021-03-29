function(test)
  set(res "")
  # Repeat "a" three times
  set(input "a")
  string_repeat("${input}" 3)
  ans(res)
  assert("${res}" STREQUAL "aaa")

  set(res "")
  # Repeat "a" three times, "-" separator
  set(input "a")
  string_repeat("${input}" 3 "-")
  ans(res)
  assert("${res}" STREQUAL "a-a-a")

  set(res "")
  # Repeat "a" three times, ";" separator makes a list
  set(input "a")
  string_repeat("${input}" 3 ";")
  ans(res)
  assert(EQUALS "${res}" "a;a;a")

  set(res "")
  # Repeat "abc" three times, ";" separator makes a list
  set(input "abc")
  string_repeat("${input}" 3 ";")
  ans(res)
  assert(EQUALS "${res}" "abc;abc;abc")

  set(res "")
  # Empty input string
  set(input "")
  string_repeat("${input}" 3 )
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # 0 repeat, returns empty string
  set(input "a")
  string_repeat("${input}" 0 )
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # 0 repeat and empty input string, returns empty string
  set(input "")
  string_repeat("${input}" 0 )
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # 1 repeat, returns original string
  set(input "a")
  string_repeat("${input}" 1)
  ans(res)
  assert("${res}" STREQUAL "a")

  set(res "")
  # 1 repeat, returns original string, separator omitted
  set(input "a")
  string_repeat("${input}" 1 ";")
  ans(res)
  assert("${res}" STREQUAL "a")
endfunction()