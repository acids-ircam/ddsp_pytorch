

# returns a copy of map with key values inverted
# only works correctly for bijective maps
function(map_invert map)
  map_keys("${map}")
  ans(keys)
  map_new()
  ans(inverted_map)
  foreach(key ${keys})
    map_tryget("${map}" "${key}")
    ans(val)
    map_set("${inverted_map}" "${val}" "${key}")
  endforeach()
  return_ref(inverted_map)
endfunction()