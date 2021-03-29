function(test)

  # clean
  memory_cache_clear("test_value")

  # get non existing cache entry
  memory_cache_get("test_value")
  ans(res)
  assert(NOT res)

  memory_cache_exists("test_value")
  ans(res)
  assert(NOT res)

  # set cache entry
  memory_cache_update("test_value" 123)
  

  memory_cache_exists("test_value")
  ans(res)
  assert(res)

  memory_cache_get("test_value")
  ans(res)
  assert("${res}" STREQUAL 123)


  # clear entry
  memory_cache_clear("test_value")
  ans(res)

  memory_cache_exists(test_value)
  ans(res)
  assert(NOT res)


function(cached_get)

  memory_cache_return_hit("myval")
set(len 2)
  map()
  foreach(i RANGE ${len})

    map("map_${i}")

    foreach(j RANGE ${len})
      kv("k1" "v1")
      kv("k2" "v2")
    endforeach()
    end()
  endforeach()

  end()
  ans(val)
  memory_cache_update("myval" "${val}")
  return_ref(val)

endfunction()
  foreach(i RANGE 100)
cached_get()

  endforeach()
endfunction()