

function(map_pop_back map prop)
  map_tryget("${map}" "${prop}")
  ans(lst)
  list_pop_back(lst)
  ans(res)
  map_set("${map}" "${prop}" ${lst})
  return_ref(res) 
endfunction()