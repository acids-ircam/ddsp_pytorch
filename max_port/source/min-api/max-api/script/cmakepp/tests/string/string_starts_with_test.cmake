function(test)
  set(res "")
  # Starts with w
  set(input "word")
  string_starts_with("${input}" "w")
  ans(res)
  assert("${res}")

  set(res "")
  # Starts with complete word
  set(input "word")
  string_starts_with("${input}" "word")
  ans(res)
  assert("${res}")

  set(res "")
  # Search is longer than input
  set(input "word")
  string_starts_with("${input}" "words")
  ans(res)
  assert(NOT "${res}")

  set(res "")
  # Does not start with search
  set(input "word")
  string_starts_with("${input}" "ord")
  ans(res)
  assert(NOT "${res}")

  set(res "")
  # Empty search
  set(input "word")
  string_starts_with("${input}" "")
  ans(res)
  assert("${res}")

  set(res "")
  # Empty input
  set(input "")
  string_starts_with("${input}" "w")
  ans(res)
  assert(NOT "${res}")

  set(res "")
  # Empty input and search
  set(input "")
  string_starts_with("${input}" "")
  ans(res)
  assert("${res}")
endfunction()