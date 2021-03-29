function(test)

  

  json_string_to_cmake("\";\"")
  ans(res)
  string(REPLACE ";" "1" res "${res}")
  assert("${res}" STREQUAL "1")

  json_string_to_cmake("\"\"")
  ans(res)
  assert("${res}_" STREQUAL "_")

  string(ASCII 10 char)
  json_string_to_cmake("\"\\n\"")
  ans(res)
  assert("${res}" STREQUAL "${char}")
    
  string(ASCII 9 char)
  json_string_to_cmake("\"\\t\"")
  ans(res)
  assert("${res}" STREQUAL "${char}")

  string(ASCII 13 char)
  json_string_to_cmake("\"\\r\"")
  ans(res)
  assert("${res}" STREQUAL "${char}")

  string(ASCII 12 char)
  json_string_to_cmake("\"\\f\"")
  ans(res)
  assert("${res}" STREQUAL "${char}")

  string(ASCII 8 char)
  json_string_to_cmake("\"\\b\"")
  ans(res)
  assert("${res}" STREQUAL "${char}")


  json_string_to_cmake("\"\\\\\"")
  ans(res)
  string(REPLACE "\\" "1" res "${res}")
  assert("${res}" STREQUAL "1")


  json_string_to_cmake("\"\\\"\"")
  ans(res)
  assert("${res}" STREQUAL "\"")



  string(ASCII 8 bs)
  string(ASCII 12 ff)
  string(ASCII 13 cr)
  string(ASCII 10 lf)
  string(ASCII 9 ht)
  json_string_to_cmake("\"\\\"\\\\\\b\\f\\r\\t\\n;a\"")
  ans(res)
  string(REPLACE ";" "1" res "${res}")
  assert("${res}" STREQUAL "\"\\${bs}${ff}${cr}${ht}${lf}1a")
endfunction()