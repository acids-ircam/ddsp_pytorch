## `(<map> <key> <any...>)-><any...>`
##
## returns the value stored in map.key or 
## sets the value at map.key to ARGN and returns 
## the value
function(map_get_default map key)
  map_has("${map}" "${key}")
  ans(has_key)
  if(NOT has_key)
    map_set("${map}" "${key}" "${ARGN}")
  endif()
  map_tryget("${map}" "${key}")
  return_ans()
endfunction()