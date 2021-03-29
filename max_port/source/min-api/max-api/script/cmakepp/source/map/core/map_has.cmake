


function(map_has this key )  
  get_property(res GLOBAL PROPERTY "${this}.${key}" SET)
  return(${res})
endfunction()

# faster way of accessing map.  however fails if key contains escape sequences, escaped vars or @..@ substitutions
# if thats the case comment out this macro
macro(map_has this key )  
  get_property(__ans GLOBAL PROPERTY "${this}.${key}" SET)
endmacro()



