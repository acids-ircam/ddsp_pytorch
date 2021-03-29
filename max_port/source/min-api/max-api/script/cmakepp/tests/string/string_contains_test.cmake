function(test)
  set(res "")
  # Empty string -> true
  string_contains("" "")
  ans(res)
  assert(res)

  set(res "")
  # Empty string, non-empty search -> false
  string_contains("" "asd")
  ans(res)
  assert(NOT res)

  set(res "")
  # Empty string, non-empty search -> true
  string_contains("asd" "")
  ans(res)
  assert(res)

  set(res "")
  # String ends with df -> true
  string_contains("asdf" "df")
  ans(res)
  assert(res)

  set(res "")
  # String does not end with mu -> false
  string_contains("asdf" "mu")
  ans(res)
  assert(NOT res)

  set(res "")
  # Long search string -> false
  string_contains("string" "longsearchstring")
  ans(res)
  assert(NOT res)

  set(res "")
  # Search string same as input -> true
  string_contains("astring" "astring")
  ans(res)
  assert(res)
  
  set(res "")
  # Contains string a -> true
  string_contains("abc" "a")
  ans(res)
  assert(res)

  set(res "")
  # Search string same as input -> true
  string_contains("abc" "d")
  ans(res)
  assert(NOT res)
endfunction()