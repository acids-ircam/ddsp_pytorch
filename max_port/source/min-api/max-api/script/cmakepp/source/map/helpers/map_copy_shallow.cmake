# copies the values of the source map into the target map by assignment
# (shallow copy)
function(map_copy_shallow target source)
  map_keys("${source}")
  ans(keys)

  foreach(key ${keys})
    map_tryget("${source}" "${key}")
    ans(val)
    map_set("${target}" "${key}" "${val}")
  endforeach()
  return()
endfunction()

