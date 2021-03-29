function(test)
  set(str "hello")

  set(res "")
  # Second char is set from "e" to "a"
  string_char_at_set(${str} 1 a)
  ans(res)
  assert("${res}" STREQUAL "hallo")

  set(res "")
  # Test for negative indexing
  # index of -2 is last char of string
  string_char_at_set(${str} -2 a)
  ans(res)
  assert("${res}" STREQUAL "hella")

  set(res "")
  # Out of bounds positive index
  string_char_at_set(${str} 6 a)
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Out of bounds negative index (1/2)
  string_char_at_set(${str} -7 a)
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Out of bounds negative index (2/2)
  string_char_at_set(${str} -1 a)
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(str "hello three words")

  set(res "")
  # Test for negative indexing
  # Multi word string
  string_char_at_set(${str} -2 e)
  ans(res)
  assert("${res}" STREQUAL "hello three worde")

  set(res "")
  # Test to set whitespace char to "s"
  string_char_at_set(${str} 5 s)
  ans(res)
  assert("${res}" STREQUAL "hellosthree words")

  set(res "")
  # Set first char of input string to empty char (delete)
  string_char_at_set(${str} 0 "")
  ans(res)
  assert("${res}" STREQUAL "ello three words")

  set(res "")
  # Set last char of input string to empty char (delete)
  string_char_at_set(${str} -2 "")
  ans(res)
  assert("${res}" STREQUAL "hello three word")

  set(res "")
  # Set an inner char of input string to empty char (delete)
  string_char_at_set(${str} 5 "")
  ans(res)
  assert("${res}" STREQUAL "hellothree words")
endfunction()
