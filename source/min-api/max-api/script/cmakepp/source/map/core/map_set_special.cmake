
function(map_set_special map key)
  set_property(GLOBAL PROPERTY "${map}.__${key}__" "${ARGN}")
  #map_set_hidden("${map}" "__${key}__" "${ARGN}")
endfunction()