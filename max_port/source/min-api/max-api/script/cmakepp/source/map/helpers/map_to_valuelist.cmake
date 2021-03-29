

  function(map_to_valuelist map)
    set(keys ${ARGN})
    list_extract_flag(keys --all)
    ans(all)
    if(all)
      map_keys(${map})
      ans(keys)
    endif()
    set(result)

    foreach(key ${keys})
      map_tryget(${map} "${key}")
      ans(value)
      list(APPEND result "${value}")
    endforeach()
    return_ref(result)
  endfunction()