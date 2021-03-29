
# converts the varargs list of pahts to a map
function(paths_to_map )
  map_new()
  ans(map)
  foreach(path ${ARGN})
    path_to_map("${map}" "${path}")
  endforeach()
  return_ref(map)
endfunction()
