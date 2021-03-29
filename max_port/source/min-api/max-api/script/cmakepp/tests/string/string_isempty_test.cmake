function(test)
  set(res "")
  # Empty string -> true
  string_isempty("")
  ans(res)
  assert(res)

  set(res "")
  # String "false" -> not empty
  string_isempty("false")
  ans(res)
  assert(NOT res)

  set(res "")
  # String "no" -> not empty
  string_isempty("no")
  ans(res)
  assert(NOT res)

  set(res "")
  # String "abc" -> not empty
  string_isempty("abc")
  ans(res)
  assert(NOT res)

  set(res "")
  # String " " -> not empty
  string_isempty(" ")
  ans(res)
  assert(NOT res)

  set(res "")
  # String "_" -> not empty
  string_isempty("_")
  ans(res)
  assert(NOT res)
endfunction()