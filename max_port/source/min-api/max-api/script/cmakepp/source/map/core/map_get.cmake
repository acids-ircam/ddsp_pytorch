
function(map_get this key)
  set(property_ref "${this}.${key}")
  get_property(property_exists GLOBAL PROPERTY "${property_ref}" SET)
  if(NOT property_exists)
    message(FATAL_ERROR "map '${this}' does not have key '${key}'")    
  endif()
  
  get_property(property_val GLOBAL PROPERTY "${property_ref}")
  return_ref(property_val)  
endfunction()
# faster way of accessing map.  however fails if key contains escape sequences, escaped vars or @..@ substitutions
# if thats the case comment out this macro
macro(map_get __map_get_map __map_get_key)
  set(__map_get_property_ref "${__map_get_map}.${__map_get_key}")
  get_property(__ans GLOBAL PROPERTY "${__map_get_property_ref}")
  if(NOT __ans)
    get_property(__map_get_property_exists GLOBAL PROPERTY "${__map_get_property_ref}" SET)
    if(NOT __map_get_property_exists)
      json_print("${__map_get_map}")
      message(FATAL_ERROR "map '${__map_get_map}' does not have key '${__map_get_key}'")    
    endif()
  endif()  
endmacro()

