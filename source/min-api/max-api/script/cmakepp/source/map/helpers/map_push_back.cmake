

function(map_push_back map prop)
  map_tryget("${map}" "${prop}")
  ans(lst)
  list_push_back(lst ${ARGN})
  map_set("${map}" "${prop}" ${lst})
  return_ref(lst)
endfunction()