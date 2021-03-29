

function(map_pop_front map prop)
  map_tryget("${map}" "${prop}")
  ans(lst)
  list_pop_front(lst)
  ans(res)
  map_set("${map}" "${prop}" ${lst})
  return_ref(res)
endfunction()