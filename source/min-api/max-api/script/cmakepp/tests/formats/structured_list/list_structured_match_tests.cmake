function(test)

# map_pairs test
obj("{a:1,b:2}")
ans(obj)
map_pairs("${obj}")
ans(res)
assert(EQUALS ${res} a 1 b 2)


# map_invert test
obj("{a:'c',b:'d'}")
ans(obj)
map_invert("${obj}")
ans(res)
json_print(${res})
assert(DEREF "{res.c}" STREQUAL  a )
assert(DEREF "{res.d}" STREQUAL  b )


# map_pick test
obj("{a:'b',c:'d'}")
ans(obj)
map_pick("${obj}" a)
ans(res)
obj("{a:'b'}")
ans(expected)
map_equal("${res}" "${expected}")
ans(res)
assert(res)



# map_omit test
obj("{a:'b',c:'d'}")
ans(obj)
map_omit("${obj}" a)
ans(res)
obj("{c:'d'}")
ans(expected)
map_equal("${res}" "${expected}")
ans(res)
assert(res)


# map_defaults test
 obj("{a:1,b:2}")
 ans(obj)
 map_defaults("${obj}" "{a:3,c:4}")
 ans(res)
  assert("${obj}" STREQUAL "${res}")
 assert(DEREF {obj.a} STREQUAL "1")
 assert(DEREF {obj.b} STREQUAL "2")
 assert(DEREF {obj.c} STREQUAL "4")



# map_match_properties test
obj("{id:4, val:'asd'}")
ans(obj)
obj("{id:4}")
ans(pred)
map_match_properties("${obj}" "${pred}")
ans(res)
assert(res)
obj("{id:5}")
ans(pred)
map_match_properties("${obj}" "${pred}")
ans(res)
assert(NOT res)

# map map_matches test
 map_matches("{id:4}")
 ans(func)
 rcall(res = "${func}"(${obj}) )
 assert(res)

 map_matches("{id:5}")
 ans(func)
 rcall(res = "${func}"(${obj}))
 assert(NOT res)

# list_match test

  script("$a = [{id:5},{id:6},{id:7}]")
  json_print(${a})

  list_match(a "{id:'^[5-6]$'}")
  ans(res)
  json_print(${res})


endfunction()