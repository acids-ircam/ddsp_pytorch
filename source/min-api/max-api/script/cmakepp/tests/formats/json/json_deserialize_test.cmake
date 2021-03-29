function(test)

  ## empty
  json_deserialize("")
  ans(res)
  assert(NOT res)


  ## empty string
  json_deserialize("\"\"")
  ans(res)
  assert(NOT res)



  json_deserialize("\"abce\"")
  ans(res)
  assert("${res}" STREQUAL "abce")


  ## escaped double quote
  json_deserialize("\"\\\"\"")
  ans(res)
  assert("${res}" STREQUAL "\"")

  ## escape backslash quote escapes
  json_deserialize("\"\\\\\"\"")
  ans(res)
  string(REPLACE "\\" "1" res "${res}")
  assert("${res}" STREQUAL "1\"")

  ## escaped backlshash
  json_deserialize("\"\\\\\"")
  ans(res)
  string(REPLACE "\\" "1" res "${res}")
  assert("${res}" STREQUAL "1")

  ## escaped newline
  json_deserialize("\"\\n\"")
  ans(res)
  assert("${res}" STREQUAL "\n")

  ## semicolon. stays as a unit separator - this is the only way to guarantee 
  ## that lists and string containing semicolons remain separated
  json_deserialize("\"a;b\"")
  ans(res)
  string_codes()
  assert("${res}" STREQUAL "a${semicolon_code}b")

  ## check that brackets are correctly decoded
  json_deserialize("\"[]\"")
  ans(res)
  assert("${res}" STREQUAL "[]")


  ## null
  json_deserialize("null")
  ans(res)
  assert(NOT res)

  ## deserailzie number
  json_deserialize("123")
  ans(res)
  assert("${res}" STREQUAL "123")

  ## deserialize scientific number
  json_deserialize("-123.323e-210")
  ans(res)
  assert("${res}" STREQUAL "-123.323e-210")

  ## deserialize bool
  json_deserialize("true")
  ans(res)
  assert("${res}" STREQUAL "true")
  json_deserialize("false")
  ans(res)
  assert("${res}" STREQUAL "false")


  ## deserialize array
  json_deserialize("[]")
  ans(res)
  assert(NOT res)

  ## array with one value
  json_deserialize("[\"a\"]")
  ans(res)
  assert("${res}" STREQUAL "a")

  ## multi value array
  json_deserialize("[\"a\",\"b\"]")
  ans(res)
  assert(${res} EQUALS a b)


  ## nested arrays are automatically flattenend.
  json_deserialize("[\"a\",\"b\",[\"c\",\"d\"]]")
  ans(res)
  assert(${res} EQUALS a b c d)

  ## empty objet
  json_deserialize("{}")
  ans(res)
  assert(res)
  map_keys(${res})
  ans(keys)
  assert(NOT keys)

  ## one key object
  json_deserialize("{\"key\":\"value\"}")
  ans(res)
  map_keys("${res}")
  ans(keys)
  assert("${keys}" STREQUAL "key")
  assertf("{res.key}" STREQUAL "value")


  ## nested object
  json_deserialize("{\"a\":{\"b\":1}}")
  ans(res)
  assertf("{res.a.b}" STREQUAL "1")


  ## array in object
  json_deserialize("{\"a\":[1,2,3]}")
  ans(res)
  assertf({res.a} EQUALS 1 2 3)



  ## two key object
  json_deserialize("{\"a\":1,\"b\":2}")
  ans(res)
  map_keys("${res}")
  ans(keys)
  assert(${keys} EQUALS a b)
  assertf("{res.a}" STREQUAL "1")
  assertf("{res.b}" STREQUAL "2")

  ## object in array
  json_deserialize("[1,{\"a\":\"b\"},3]")
  ans(res)
  assertf({res[0]} STREQUAL "1")
  assertf({res[1].a} STREQUAL "b")
  assertf({res[2]} STREQUAL "3")

  cmakepp_config(base_dir)
  ans(base_dir)
  fread("${base_dir}/resources/expr-definition.json")
  ans(json)

  timer_start(t1)
  json_deserialize("${json}")
  ans(res)
  timer_print_elapsed(t1)
  
  
  return()
  


return()

  json_deserialize("
    {
  \"args\": {\"asd\":\"bsd\" }, 
  \"data\": \"hello world\", 
  \"files\": { \"asd\":\"bsd\"}, 
  \"form\": { \"asd\":\"bsd\"},
  \"headers\": {
    \"Accept\": \"*/*\", 
    \"Content-Length\": \"11\", 
    \"Host\": \"httpbin.org\"
  },
  \"json\": \"dasd\", 
  \"origin\": \"85.181.212.90\", 
  \"url\": \"http://httpbin.org/put\"
}
")
  ans(res)

  return()


  json2("{\"asd\":\"he[]ll;o\", \"bsd\":[1,2,3,4]}")

  timer_start(t1)
  json2("{\"asd\":\"he[]ll;o\", \"bsd\":[1,2,3]}")
  ans(res)
  timer_print_elapsed(t1)


  timer_start(t1)
  json3("{\"a.sd\":\"he[]ll;o\", \"bs;d\":[1,[5,6],{ \"asd\":\"abc\" \"def\", \"gugugaga\":[true,false,1,null,1,null,\"lala\"] },3]}")
  ans(res)
  timer_print_elapsed(t1)
  timer_start(t1)
  json2("{\"a.sd\":\"he[]ll;o\", \"bs;d\":[1,[5,6],{ \"asd\":\"abc\" \"def\", \"gugugaga\":[true,false,1,null,1,null,\"lala\"] },3]}")
  ans(res)
  timer_print_elapsed(t1)



  
  ## issue #48
  json_deserialize("
{
  \"name\": \"library\",
  \"version\": \"1.0.0\",  
  \"platforms\": {
          \"windows-x86_64\": {
              \"env\":\"intel12.1\", \"binpath\": \"\", \"libpath\": \"win64/intel12.1\" 
          },
          \"linux-x86_64\": {
              \"env\":\"gnu4.8\", \"binpath\": \"\", \"libpath\": \"\"
          }
    }

}

    ")
  ans(res)
json_print("${res}")

endfunction()