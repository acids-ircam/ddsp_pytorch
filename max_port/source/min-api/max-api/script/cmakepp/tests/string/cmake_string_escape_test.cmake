function(test)
  cmake_string_escape("abcdefg")
  ans(res)
  assert("abcdefg" STREQUAL "${res}")


  cmake_string_escape("\\")
  ans(res)
  string(REPLACE "\\" "1" res "${res}")
  assert("${res}" STREQUAL "11")

  cmake_string_unescape("\\\\")
  ans(res)
  string(REPLACE "\\" "1" res "${res}")
  assert("${res}" STREQUAL "1")


  cmake_string_escape("\"")
  ans(res)
  assert("${res}" STREQUAL "\\\"")

  cmake_string_unescape("\\\"")
  ans(res)
  assert("${res}" STREQUAL "\"")


  cmake_string_escape("(")
  ans(res)
  assert("_${res}" STREQUAL "_\\(")

  cmake_string_unescape("\\(")
  ans(res)
  assert("_${res}" STREQUAL "_(")

  cmake_string_escape(")")
  ans(res)
  assert("_${res}" STREQUAL "_\\)")

  cmake_string_unescape("\\)")
  ans(res)
  assert("_${res}" STREQUAL "_)")

  cmake_string_escape("$")
  ans(res)
  assert("${res}" STREQUAL "\\$" )

  cmake_string_unescape("\\$")
  ans(res)
  assert("${res}" STREQUAL "$")


  cmake_string_escape("#")
  ans(res)
  assert("${res}" STREQUAL "\\#" )

  cmake_string_unescape("\\#")
  ans(res)
  assert("${res}" STREQUAL "#")


  cmake_string_escape("^")
  ans(res)
  assert("${res}" STREQUAL "\\^")

  cmake_string_unescape("\\^")
  ans(res)
  assert("${res}" STREQUAL "^")

  set(complex "\\!@#$%^ &*()__)( *&^%$\"# \" \ @!   ;'\\'][[]/.,.,m,[!@#$%^&*()_++_)(*&^%$#@!]{}]]][[][]{}{}\\|###")

  cmake_string_escape("${complex}")
  ans(res1)
  cmake_string_unescape("${res1}")
  ans(res)
  cmake_string_escape("${res}")
  ans(res2)
 string(COMPARE EQUAL "${res1}" "${res2}" res)
 assert(res)
endfunction()