function(test)
  set(res "")
  # Empty string not a number
  string_isnumeric("")
  ans(res)
  assert(NOT res)

  set(res "")
  # String not a number
  string_isnumeric("asva")
  ans(res)
  assert(NOT res)

  set(res "")
  # Is a number
  string_isnumeric("123")
  ans(res)
  assert(res)

  set(res "")
  # Zero is a number
  string_isnumeric(0)
  ans(res)
  assert(res)

  set(res "")
  # One is a number
  string_isnumeric(1)
  ans(res)
  assert(res)

  set(res "")
  # Nine is a number
  string_isnumeric(9)
  ans(res)
  assert(res)

  set(res "")
  # Ten is a number
  string_isnumeric(10)
  ans(res)
  assert(res)

  set(res "")
  # Long number
  string_isnumeric("1231023")
  ans(res)
  assert(res)
endfunction()