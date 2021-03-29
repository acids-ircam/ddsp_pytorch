function(test)
  # Remove delimiter ' of a string
  set(str "'c:\\a b c\\d e f'")
  string_take_delimited(str ')
  ans(res)
  assert("${res}" STREQUAL "c:\\a b c\\d e f")
  assert("${str}_" STREQUAL "_")

  # Only first and last delimiter ' of a string are removed.
  set(str "'c:\\a b c\\d e \\'f'")
  string_take_delimited(str ')
  ans(res)
  assert("${res}" STREQUAL "c:\\a b c\\d e 'f")
  assert("${str}_" STREQUAL "_")

  # Empty string
  set(str "")
  string_take_delimited(str)
  ans(res)
  assert("${res}_" STREQUAL "_")
  assert("${str}_" STREQUAL "_")

  # Delimiter only string
  set(str "''")
  string_take_delimited(str ')
  ans(res)
  assert("${res}_" STREQUAL "_")
  assert("${str}_" STREQUAL "_")

  # First element of a list of strings is extracted and the delimiter removed
  set(str "'a','b'")
  string_take_delimited(str "''")
  ans(res)
  assert("${res}" STREQUAL "a")
  assert("${str}" STREQUAL ",'b'")

  # Arbritary delimiter can be set
  set(str "<asdas>")
  string_take_delimited(str "<>")
  ans(res)
  assert("${res}" STREQUAL "asdas")
  assert("${str}_" STREQUAL "_")

  # Default delimiter is "". First element of a list is extracted and returned
  set(str "\"wininit.exe\",\"480\",\"Services\",\"0\",\"616 K\",\"Unknown\",\"N/A\",\"0:00:00\",\"N/A\"")
  string_take_delimited(str)  
  ans(res)
  assert("${res}" STREQUAL "wininit.exe")
  assert("${str}" STREQUAL ",\"480\",\"Services\",\"0\",\"616 K\",\"Unknown\",\"N/A\",\"0:00:00\",\"N/A\"")
endfunction()