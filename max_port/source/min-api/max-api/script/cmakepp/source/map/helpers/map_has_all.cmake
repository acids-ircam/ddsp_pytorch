
# returns true if map has all keys specified
#as varargs
function(map_has_all map)

  foreach(key ${ARGN})
    map_has("${map}" "${key}")
    ans(has_key)
    if(NOT has_key)
      return(false)
    endif()
  endforeach()
  return(true)

endfunction()
