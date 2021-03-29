
function(address_peek_front ref)
  address_get(${ref})
  ans(value)
  list_peek_front(value ${ARGN})
  ans(res)
  return_ref(res)
endfunction()