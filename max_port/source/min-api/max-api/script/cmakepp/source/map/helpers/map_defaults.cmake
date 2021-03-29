
# sets all undefined properties of map to the default value
function(map_defaults map defaults)
  obj("${defaults}")
  ans(defaults)
  if(NOT defaults)
    message(FATAL_ERROR "No defaults specified")
  endif()

  if(NOT map)
    map_new()
    ans(map)
  endif()

  map_keys("${map}")
  ans(keys)

  map_keys("${defaults}")
  ans(default_keys)


  if(default_keys AND keys)
    list(REMOVE_ITEM default_keys ${keys})
  endif()
  foreach(key ${default_keys})
    map_tryget("${defaults}" "${key}")
    ans(val)
    map_set("${map}" "${key}" "${val}")
  endforeach()
  return_ref(map)
endfunction()