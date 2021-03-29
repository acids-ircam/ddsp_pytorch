## 
## imports all properties of map into local scope
function(map_import_properties_all __map)
  if(NOT __map)
    return()
  endif()
  map_keys("${__map}")
  ans(__mipa_keys)
  foreach(key ${__mipa_keys})
    map_tryget("${__map}" "${key}")
    set("${key}" "${__ans}" PARENT_SCOPE)
  endforeach()
  return()
endfunction()