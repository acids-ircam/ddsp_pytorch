
# returns true if map has any of the keys
# specified as varargs
function(map_has_any map)
  foreach(key ${ARGN})
    map_has("${map}" "${key}")
    ans(has_key)
    if(has_key)
      return(true)
    endif()
  endforeach()
  return(false)

endfunction()