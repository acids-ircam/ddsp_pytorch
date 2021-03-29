## renames a key in the specified map
function(map_rename map key_old key_new)
  map_get("${map}" "${key_old}")
  ans(value)
  map_remove("${map}" "${key_old}")
  map_set("${map}" "${key_new}" "${value}")
endfunction()
