function(test) 
  # "lhs " should be removed in both strings
  set(in_lhs "lhs test")
  set(in_rhs "lhs a")
  string_trim_to_difference(in_lhs in_rhs)
  assert("${in_lhs}" STREQUAL "test")
  assert("${in_rhs}" STREQUAL "a")

  # Same string content results in empty string
  set(in_lhs "lhs")
  set(in_rhs "lhs")
  string_trim_to_difference(in_lhs in_rhs)
  assert("${in_lhs}_" STREQUAL "_")
  assert("${in_rhs}_" STREQUAL "_")

  # One empty string causes no change (1/2)
  set(in_lhs "")
  set(in_rhs "lhs a")
  string_trim_to_difference(in_lhs in_rhs)
  assert("${in_lhs}_" STREQUAL "_")
  assert("${in_rhs}" STREQUAL "lhs a")

  # One empty string causes no change (2/2)
  set(in_lhs "lhs a")
  set(in_rhs "")
  string_trim_to_difference(in_lhs in_rhs)
  assert("${in_lhs}" STREQUAL "lhs a")
  assert("${in_rhs}_" STREQUAL "_")

  # Two empty strings are still empty
  set(in_lhs "")
  set(in_rhs "")
  string_trim_to_difference(in_lhs in_rhs)
  assert("${in_lhs}_" STREQUAL "_")
  assert("${in_rhs}_" STREQUAL "_")

  # Two strings with different beginning are unchanged
  set(in_lhs "same test")
  set(in_rhs "diff test")
  string_trim_to_difference(in_lhs in_rhs)
  assert("${in_lhs}" STREQUAL "same test")
  assert("${in_rhs}" STREQUAL "diff test")
endfunction()