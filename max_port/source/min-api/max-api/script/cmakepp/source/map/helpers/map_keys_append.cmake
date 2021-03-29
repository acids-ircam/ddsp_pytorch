
function(map_keys_append map)
  set_property(GLOBAL APPEND PROPERTY "${map}" ${ARGN})
endfunction()