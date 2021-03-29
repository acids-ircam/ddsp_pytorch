function(test)
  # Empty string
  set(input "")
  string_split_at_last(partA partB "${input}" "@")
  assert(NOT partA)
  assert(NOT partB)

  # Empty split char, nothing is split
  set(input "a@b")
  string_split_at_last(partA partB "${input}" "")
  assert("${partA}" STREQUAL "a@b")
  assert(NOT partB)

  # Empty input and split char
  set(input "")
  string_split_at_last(partA partB "${input}" "")
  assert(NOT partA)
  assert(NOT partB)

  # Middle split
  set(input "a@b")
  string_split_at_last(partA partB "${input}" "@")
  assert("${partA}" STREQUAL "a")
  assert("${partB}" STREQUAL "b")

  # Two split chars
  set(input "a@b@c")
  string_split_at_last(partA partB "${input}" "@")
  assert("${partA}" STREQUAL "a@b")
  assert("${partB}" STREQUAL "c")

  # Nothing to split
  set(input "word")
  string_split_at_last(partA partB "${input}" "@")
  assert("${partA}" STREQUAL "word")
  assert(NOT partB)

  # Split at beginning, no partA
  set(input "@end")
  string_split_at_last(partA partB "${input}" "@")
  assert(NOT partA)
  assert("${partB}" STREQUAL "end")

  # Split at end, no partB
  set(input "beginning@")
  string_split_at_last(partA partB "${input}" "@")
  assert("${partA}" STREQUAL "beginning")
  assert(NOT partB)
endfunction()