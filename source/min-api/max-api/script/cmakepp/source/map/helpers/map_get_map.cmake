## `(<map> <key>)-><map>`
##
## returns a map for the specified key
## creating it if it does not exist
##
function(map_get_map map key)
  map_tryget(${map} ${key})
  ans(res)
  is_address("${res}")
  ans(ismap)
  if(NOT ismap)
    map_new()
    ans(res)
    map_set(${map} ${key} ${res})
  endif()
  return_ref(res)
endfunction()

