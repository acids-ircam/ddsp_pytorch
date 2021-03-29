function(test)

cmake_string_to_json("a;a")
ans(res)
string(REPLACE "\\\\" "1" res "${res}")
string(REPLACE ";" "2" res "${res}")
assert("${res}" STREQUAL "\"a12a\"")

cmake_string_to_json("")
ans(res)
assert("${res}" STREQUAL "\"\"")

string(ASCII 10 char)
cmake_string_to_json("${char}")
ans(res)
assert("${res}" STREQUAL "\"\\n\"")
  
string(ASCII 9 char)
cmake_string_to_json("${char}")
ans(res)
assert("${res}" STREQUAL "\"\\t\"")

string(ASCII 13 char)
cmake_string_to_json("${char}")
ans(res)
assert("${res}" STREQUAL "\"\\r\"")

string(ASCII 12 char)
cmake_string_to_json("${char}")
ans(res)
assert("${res}" STREQUAL "\"\\f\"")

string(ASCII 8 char)
cmake_string_to_json("${char}")
ans(res)
assert("${res}" STREQUAL "\"\\b\"")

cmake_string_to_json("\\")
ans(res)
assert("${res}" STREQUAL "\"\\\\\"")


cmake_string_to_json("\"")
ans(res)
assert("${res}" STREQUAL "\"\\\"\"")



  string(ASCII 8 bs)
  string(ASCII 12 ff)
  string(ASCII 13 cr)
  string(ASCII 10 lf)
  string(ASCII 9 ht)
  set(str "\\\"\\${bs}${ff}${cr}${ht}${lf};a")
  
  cmake_string_to_json("${str}")
  ans(res)
  string(REPLACE ";" "1" res "${res}")
  assert("${res}" STREQUAL "\"\\\\\\\"\\\\\\b\\f\\r\\t\\n\\\\1a\"")

endfunction()