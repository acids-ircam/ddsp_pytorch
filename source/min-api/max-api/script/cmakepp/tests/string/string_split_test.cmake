function(test)
  set(res "")
  # Empty string
  set(input "")
  string_split("${input}" "@")
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Nothing is split
  set(input "a@b")
  string_split("${input}" "c")
  ans(res)
  assert("${res}" STREQUAL "a@b")

  set(res "")
  # List of two empty strings (1/2)
  set(input "@")
  string_split("${input}" "@")
  ans(res)
  assert(EQUALS "${res}" ";")

  set(res "")
  # List of two empty strings (2/2)
  set(input "\n")
  string_split("${input}" "\n")
  ans(res)
  assert(EQUALS "${res}" ";")

  set(res "")
  # Middle split
  set(input "a@b")
  string_split("${input}" "@")
  ans(res)
  set(expected a b)
  #assert_list_equal(res expected)
  assert(EQUALS "${res}" "${expected}")
  
  set(res "")
  # Two split chars
  set(input "a@b@c")
  string_split("${input}" "@")
  ans(res)
  set(expected a b c)
  assert(EQUALS "${res}" "${expected}")

  set(res "")
  # Nothing to split
  set(input "word")
  string_split("${input}" "@")
  ans(res)
  assert(NOT ${res})

  set(res "")
  # Split at beginning
  set(input "@end")
  string_split("${input}" "@")
  ans(res)
  set(expected "end")
  assert(EQUALS "${res}" "${expected}")

  set(res "")
  # Split at end
  set(input "beginning@")
  string_split("${input}" "@")
  ans(res)
  set(expected "beginning")
  assert(EQUALS "${res}" "${expected}")
endfunction()