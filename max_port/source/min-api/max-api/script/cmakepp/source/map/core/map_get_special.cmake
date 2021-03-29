
function(map_get_special map key)
  map_tryget("${map}" "__${key}__")
  return_ans()
endfunction()

## faster
macro(map_get_special map key)
  get_property(__ans GLOBAL PROPERTY "${map}.__${key}__")
endmacro()

## correcter
function(map_get_special map key)
  get_property(__ans GLOBAL PROPERTY "${map}.__${key}__")
  set(__ans "${__ans}" PARENT_SCOPE)
endfunction()
