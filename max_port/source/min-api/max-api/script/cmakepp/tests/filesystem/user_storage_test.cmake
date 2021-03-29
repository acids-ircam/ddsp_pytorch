function(test)

  ## print the user data path
  user_data_dir()
  ans(res)
  message("user data is located in ${res}")

  ## remove any prexistin user data under testkey
  user_data_clear("testkey")

  # read empty user data
  user_data_read("testkey")
  ans(res)
  assert("${res}_" STREQUAL "_")


  # write simple user_data
  user_data_write("testkey" hello)
  ans(res)
  assert(EXISTS "${res}")

  user_data_read("testkey")
  ans(res)
  assert("${res}" STREQUAL "hello")


  # read/write complex user_data
  map()
    kv("h1" h1)
    kv("h2" h1)
    kv("h3" h1)
    kv("h4" h1)
    map(43)
      kv(jasd  asd)
    end()

  end()
  ans(res)
  set(original "${res}")
  user_data_write(testkey "${res}")

  user_data_read(testkey)
  ans(res2)

  map_equal("${res}" "${res2}")
  ans(isequal)

  assert(isequal)
  assert(NOT "${res}" STREQUAL "${res2}")


  ## get stored user data keys
  user_data_ids()
  ans(res)

  assert(CONTAINS testkey ${res})

  ## get a single user data
  user_data_get(testkey 43.jasd)
  ans(res)
  assert("${res}" STREQUAL "asd")


  ## get root user data
  user_data_get(testkey)
  ans(res)
  assert(res)
  map_equal("${res}" "${original}")
  ans(isequal)

  assert(isequal)


  ## set specific user data
  user_data_set(testkey a.b.c 323)
  user_data_get(testkey a.b.c)
  ans(res)
  assert("${res}" STREQUAL 323)

  ## set root user data
  user_data_set(testkey . 123)
  user_data_get(testkey)
  ans(res)
  assert("${res}" STREQUAL "123")





endfunction()