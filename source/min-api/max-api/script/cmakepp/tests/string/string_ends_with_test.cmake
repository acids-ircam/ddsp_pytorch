function(test)
  set(res "")
  # Empty string -> true
  string_ends_with("" "")
  ans(res)
  assert(res)

  set(res "")
  # Empty string, non-empty search -> false
  string_ends_with("" "asd")
  ans(res)
  assert(NOT res)

  set(res "")
  # Empty string, non-empty search -> true
  string_ends_with("asd" "")
  ans(res)
  assert(res)

  set(res "")
  # String ends with df -> true
  string_ends_with("asdf" "df")
  ans(res)
  assert(res)

  set(res "")
  # String does not end with mu -> false
  string_ends_with("asdf" "mu")
  ans(res)
  assert(NOT res)

  set(res "")
  # Long search string -> false
  string_ends_with("string" "longsearchstring")
  ans(res)
  assert(NOT res)

  set(res "")
  # Search string same as input -> true
  string_ends_with("astring" "astring")
  ans(res)
  assert(res)
endfunction()