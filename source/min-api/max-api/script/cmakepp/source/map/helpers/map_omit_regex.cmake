# returns a map with all properties except those matched by any of the specified regexes
function(map_omit_regex map)
  set(regexes ${ARGN})
  map_keys("${map}")
  ans(keys)

  foreach(regex ${regexes})
    foreach(key ${keys})
        if("${key}" MATCHES "${regex}")
          list_remove(keys "${key}")
        endif()
    endforeach()
  endforeach()


  map_pick("${map}" ${keys})

  return_ans()


endfunction()