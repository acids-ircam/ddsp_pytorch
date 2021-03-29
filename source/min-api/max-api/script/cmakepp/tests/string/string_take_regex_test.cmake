function(test)  
  set(res "")
  # Whitespaces at beginning are removed
  set(input "  whitespaces")
  string_take_regex(input "[ ]*" "")
  ans(res)
  assert("${res}" STREQUAL "  ")
  assert("${input}" STREQUAL "whitespaces")
  
  set(res "")
  # Whitespaces at the end are not removed
  set(input " whitespaces  ")
  string_take_regex(input "[ ]*" "")
  ans(res)
  assert("${res}" STREQUAL " ")
  assert("${input}" STREQUAL "whitespaces  ")

  set(res "")
  # Whitespaces only at beginning are removed
  set(input "  whitespaces  ")
  string_take_regex(input "[ ]*" "")
  ans(res)
  assert("${res}" STREQUAL "  ")
  assert("${input}" STREQUAL "whitespaces  ")

  set(res "")
  # Whitespaces in the middle are kept
  set(input "  white  spaces")
  string_take_regex(input "[ ]*" "")
  ans(res)
  assert("${res}" STREQUAL "  ")
  assert("${input}" STREQUAL "white  spaces")

  set(res "")
  # Whitespaces only result in empty string "input" (1/2)
  set(input " ")
  string_take_regex(input "[ ]*" "")
  ans(res)
  assert("${res}" STREQUAL " ")
  assert("${input}_" STREQUAL "_")

  set(res "")
  # Whitespaces only result in empty string "input" (2/2)
  set(input "   ")
  string_take_regex(input "[ ]*" "")
  ans(res)
  assert("${res}" STREQUAL "   ")
  assert("${input}_" STREQUAL "_")

  set(res "")
  # Removes first 3 chars
  set(input "abcd")
  string_take_regex(input "abc" "")
  ans(res)
  assert("${res}" STREQUAL "abc")
  assert("${input}" STREQUAL "d")
endfunction()