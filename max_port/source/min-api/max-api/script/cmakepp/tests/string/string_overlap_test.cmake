function(test)
  set(res "")
  # Empty input
  set(input1 "")
  set(input2 "")
  string_overlap("${input1}" "${input2}")
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Empty input2
  set(input1 "abc")
  set(input2 "")
  string_overlap("${input1}" "${input2}")
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Empty input1
  set(input1 "")
  set(input2 "abc")
  string_overlap("${input1}" "${input2}")
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # One char overlap
  set(input1 "abc")
  set(input2 "axy")
  string_overlap("${input1}" "${input2}")
  ans(res)
  assert("${res}" STREQUAL "a")

  set(res "")
  # No overlap
  set(input1 "zabc")
  set(input2 "yabc")
  string_overlap("${input1}" "${input2}")
  ans(res)
  assert("${res}_" STREQUAL "_")

  set(res "")
  # Complete overlap
  set(input1 "abc")
  set(input2 "abc")
  string_overlap("${input1}" "${input2}")
  ans(res)
  assert("${res}" STREQUAL "abc")

  set(res "")
  # Whitespace overlap
  set(input1 " ")
  set(input2 " ")
  string_overlap("${input1}" "${input2}")
  ans(res)
  assert("${res}" STREQUAL " ")
endfunction()