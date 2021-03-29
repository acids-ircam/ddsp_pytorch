# tries to get the value map[key] and returns NOTFOUND if
# it is not found

function(map_tryget map key)
  get_property(res GLOBAL PROPERTY "${map}.${key}")
  return_ref(res)
endfunction()

# faster way of accessing map.  however fails if key contains escape sequences, escaped vars or @..@ substitutions
# if thats the case comment out this macro
macro(map_tryget map key)
  get_property(__ans GLOBAL PROPERTY "${map}.${key}")
endmacro()