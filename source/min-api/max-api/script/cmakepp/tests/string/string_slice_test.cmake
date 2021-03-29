function(test)
  set(res "")
  # Empty string, index 0 for slice
  set(input "")
  string_slice("${input}" 0 0)
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Index 0 for slice
  set(input "word")
  string_slice("${input}" 0 0)
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # First char extracted
  set(input "word")
  string_slice("${input}" 0 1)
  ans(res)
  assert("${res}" STREQUAL "w")

  set(res "")
  # First two chars extracted
  set(input "word")
  string_slice("${input}" 0 2)
  ans(res)
  assert("${res}" STREQUAL "wo")

  set(res "")
  # Whole word slice
  set(input "word")
  string_slice("${input}" 0 4)
  ans(res)
  assert("${res}" STREQUAL "word")

  set(res "")
  # Whole word slice, negative indexing
  set(input "word")
  string_slice("${input}" -5 4)
  ans(res)
  assert("${res}" STREQUAL "word")

  set(res "")
  # First char, negative indexing
  set(input "word")
  string_slice("${input}" -5 1)
  ans(res)
  assert("${res}" STREQUAL "w")

  set(res "")
  # Last char, negative indexing
  set(input "word")
  string_slice("${input}" -2 4)
  ans(res)
  assert("${res}" STREQUAL "d")

  set(res "")
  # Empty return
  set(input "word")
  string_slice("${input}" 4 4)
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Last char
  set(input "word")
  string_slice("${input}" 3 4)
  ans(res)
  assert("${res}" STREQUAL "d")
endfunction()