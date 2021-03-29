
function(address_pop_back ref)
  address_get(${ref})
  ans(value)
  list_pop_back(value)
  ans(res)
  address_set(${ref} ${value})
  return_ref(res)
endfunction()