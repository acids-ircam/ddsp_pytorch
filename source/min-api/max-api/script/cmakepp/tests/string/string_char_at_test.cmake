function(test)

  set(str "hello")

  set(res "")
  # Second char is "e"
  string_char_at(${str} 1)
  ans(res)
  assert("${res}" STREQUAL "e")

  set(res "")
  # Test for negative indexing
  # index of -2 is last char of string
  string_char_at(${str} -2)
  ans(res)
  assert("${res}" STREQUAL "o")

  set(res "")
  # Out of bounds positive index
  string_char_at(${str} 6)
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Out of bounds negative index (1/2)
  string_char_at(${str} -7)
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Out of bounds negative index (2/2)
  string_char_at(${str} -1)
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(str "hello three words")

  set(res "")
  # Test for negative indexing
  # Multi word string
  string_char_at(${str} -2)
  ans(res)
  assert("${res}" STREQUAL "s")

  set(res "")
  # Test to get whitespace char
  string_char_at(${str} 5)
  ans(res)
  assert("${res}" STREQUAL " ")

  set(res "")
  # Get first char of input string
  string_char_at(${str} 0)
  ans(res)
  assert("${res}" STREQUAL "h")

  set(res "")
  # Get last char of input string
  string_char_at(${str} -2 "")
  ans(res)
  assert("${res}" STREQUAL "s") 
endfunction()