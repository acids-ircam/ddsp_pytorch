

  # creates a new map only getting the specified keys
  function(obj_pick map)
    map_new()
    ans(res)
    foreach(key ${ARGN})
      obj_get(${map} "${key}")
      ans(val)

      map_set("${res}" "${key}" "${val}")
    endforeach()
    return("${res}")
  endfunction()