function(test)
  set(res "")
  # Empty input
  set(input "")
  string_regex_escape("${input}")
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # No chars to be escaped
  set(input "@{test!}")
  string_regex_escape("${input}")
  ans(res)
  assert("${res}" STREQUAL "@{test!}")

  set(res "")
  # / is escaped
  set(input "/")
  string_regex_escape("${input}")
  ans(res)
  assert("${res}" STREQUAL "\\/")

  set(res "")
  # Escape brackets []
  set(input "[]")
  string_regex_escape("${input}")
  ans(res)
  assert("${res}" STREQUAL "\\[\\]")

  set(res "")
  # Escape brackets ()
  set(input "()")
  string_regex_escape("${input}")
  ans(res)
  assert("${res}" STREQUAL "\\(\\)")

  set(res "")
  # Escape test for all chars except "\"
  set(input "/ ] [ * . - ^ $ ? ) ( |")
  string_regex_escape("${input}")
  ans(res)
  assert("${res}" STREQUAL "\\/ \\] \\[ \\* \\. \\- \\^ \\$ \\? \\) \\( \\|")
endfunction()