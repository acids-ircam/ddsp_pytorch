
function(address_pop_front ref)
  address_get(${ref})
  ans(value)
  list_pop_front(value)
  ans(res)
  address_set(${ref} ${value})
  return_ref(res)
endfunction()