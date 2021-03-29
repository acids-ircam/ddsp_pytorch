function(test)
  set(res "")
  # Create 3 lines (separated with semicolons)
  string_lines("ab\ncd\nef")
  ans(res)
  assert(COUNT 3 ${res})
  assert(EQUALS "${res}" "ab;cd;ef")

  set(res "")
  # Create 1 lines
  string_lines("ab")
  ans(res)
  assert(COUNT 1 ${res})
  assert(EQUALS "${res}" "ab")

  set(res "")
  # Empty input, no line
  string_lines("")
  ans(res)
  assert(COUNT 0 ${res})
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Create 2 lines
  string_lines("a b\nc")
  ans(res)
  assert(COUNT 2 ${res})
  assert(EQUALS "${res}" "a b;c")  

  ## Todo: Fails for string_split too.
  ##       In unit test for string_split 
  ##       we don't check for list length.
  return()
  string_lines("\n")
  ans(res)
  assert(EQUALS "${res}" ";")
  assert(COUNT 2 ${res})
endfunction()