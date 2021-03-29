function(test)

  # clean
  file_cache_clear("test_value")

  # get non existing cache entry
  file_cache_get("test_value")
  ans(res)
  assert(NOT res)

  file_cache_exists("test_value")
  ans(res)
  assert(NOT res)

  # set cache entry
  file_cache_update("test_value" 123)
  

  file_cache_exists("test_value")
  ans(res)
  assert(res)

  file_cache_get("test_value")
  ans(res)
  assert("${res}" STREQUAL 123)


  # clear entry
  file_cache_clear("test_value")
  ans(res)

  file_cache_exists(test_value)
  ans(res)
  assert(NOT res)

return()

function(cached_get)

  file_cache_return_hit("myval")
set(len 30)
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
  file_cache_update("myval" "${val}")
  return_ref(val)

endfunction()
  foreach(i RANGE 10)
cached_get()

  endforeach()

endfunction()