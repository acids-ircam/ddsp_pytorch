function(test)



  key_value_store(checksum_string "kv1")
  ans(store)

  assign(allkeys = store.keys())
  assert(NOT allkeys)

  assign(key = store.save("hello world!"))
  assert(key)

  assign(value = store.load("${key}"))
  assert("${value}" STREQUAL "hello world!")

  assign(allkeys = store.keys())
  assert(allkeys)

  assert("${allkeys}" STREQUAL "${key}")


  assign(success = store.delete("${key}"))
  assert(success)


  assign(allkeys = store.keys())

  assert("${allkeys}_" STREQUAL "_")


  assign(k1 = store.save("v1"))
  assign(k1 = store.save("v2"))
  assign(k1 = store.save("v3"))

  assign(allvals = store.list())
  assert(${allvals} CONTAINS "v1")
  assert(${allvals} CONTAINS "v2")
  assert(${allvals} CONTAINS "v3")


endfunction()