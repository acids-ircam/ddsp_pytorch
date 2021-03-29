
function(map_push_front map prop)
  map_tryget("${map}" "${prop}")
  ans(lst)
  list_push_front(lst ${ARGN})
  ans(res)
  map_set("${map}" "${prop}" ${lst})
  return_ref(res)
endfunction()
