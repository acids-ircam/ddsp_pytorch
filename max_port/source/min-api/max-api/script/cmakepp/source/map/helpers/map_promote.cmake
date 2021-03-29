
macro(map_promote __map_promote_map)
  # garbled names help free from variable collisions
  map_keys(${__map_promote_map} )
  ans(__map_promote_keys)
  foreach(__map_promote_key ${__map_promote_keys})
    map_get(${__map_promote_map}  ${__map_promote_key})
    ans(__map_promote_value)
    set("${__map_promote_key}" "${__map_promote_value}" PARENT_SCOPE)
  endforeach()
endmacro()