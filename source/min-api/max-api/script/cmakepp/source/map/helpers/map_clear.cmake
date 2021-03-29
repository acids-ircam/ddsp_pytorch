

# removes all properties from map
function(map_clear map)
  map_keys("${map}")
  ans(keys)
  foreach(key ${keys})
    map_remove("${map}" "${key}")
  endforeach()
  return()
endfunction()