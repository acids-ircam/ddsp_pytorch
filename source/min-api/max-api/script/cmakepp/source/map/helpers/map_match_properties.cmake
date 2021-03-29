# returns true if map's properties match all properties of attrs
function(map_match_properties map attrs)
  map_keys("${attrs}")
  ans(attr_keys)
  foreach(key ${attr_keys})

    map_tryget("${map}" "${key}")
    ans(val)
    map_tryget("${attrs}" "${key}")
    ans(pred)
   # message("matching ${map}'s ${key} '${val}' with ${pred}")
    if(NOT "${val}" MATCHES "${pred}")
      return(false)
    endif()
  endforeach()
  return(true)
endfunction()

