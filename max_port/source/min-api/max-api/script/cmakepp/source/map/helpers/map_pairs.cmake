

# returns a list key;value;key;value;...
# only works if key and value are not lists (ie do not contain ;)
function(map_pairs map)
  map_keys("${map}")
  ans(keys)
  set(pairs)
  foreach(key ${keys})
    map_tryget("${map}" "${key}")
    ans(val)
    list(APPEND pairs "${key}")
    list(APPEND pairs "${val}")
  endforeach()
  return_ref(pairs)
endfunction()
