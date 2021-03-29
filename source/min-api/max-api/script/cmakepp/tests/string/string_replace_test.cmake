function(test)
  set(res "")
  # Replace first char with "z"
  set(input "abc")
  string_replace("${input}" "a" "z")
  ans(res)
  assert("${res}" STREQUAL "zbc")

  set(res "")
  # Empty input
  set(input "")
  string_replace("${input}" "a" "z")
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Empty replace -> removes char
  set(input "abc")
  string_replace("${input}" "a" "")
  ans(res)
  assert("${res}" STREQUAL "bc")

  set(res "")
  # Empty search -> original string returned
  set(input "abc")
  string_replace("${input}" "" "z")
  ans(res)
  assert("${res}" STREQUAL "abc")

  set(res "")
  # Empty search, replace -> original string returned
  set(input "abc")
  string_replace("${input}" "" "")
  ans(res)
  assert("${res}" STREQUAL "abc")

  set(res "")
  # Empty all
  set(input "")
  string_replace("${input}" "" "")
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Two chars, both replaced
  set(input "aac")
  string_replace("${input}" "a" "z")
  ans(res)
  assert("${res}" STREQUAL "zzc")

  set(res "")
  # Search is two chars long
  set(input "aac")
  string_replace("${input}" "aa" "z")
  ans(res)
  assert("${res}" STREQUAL "zc")
endfunction()