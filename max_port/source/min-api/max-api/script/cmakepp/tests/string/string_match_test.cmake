function(test)
  set(res "")
  # String is found
  string_match("abc" b)
  ans(res)
  assert(res)

  set(res "")
  # String not found
  string_match("abc" d)
  ans(res)
  assert(NOT res)

  set(res "")
  # Empty string, no match
  string_match("" a)
  ans(res)
  assert(NOT res)

  set(res "")
  # Empty string, empty match string
  string_match("" "")
  ans(res)
  assert(res)

  set(res "")
  # Empty match string equals match
  string_match("asdasd" "")
  ans(res)
  assert(res)

  set(res "")
  string_match("a?" "[a-z]+\\?")
  ans(res)
  assert(res)  

  set(res "")
  string_match("a bc ." "^b")
  ans(res)
  assert(NOT res)  
endfunction()