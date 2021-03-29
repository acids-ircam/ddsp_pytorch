
  function(address_push_back ref)
    address_get(${ref})
    ans(value)
    list_push_back(value "${ARGN}")
    ans(res)
    address_set(${ref} ${value})
    return_ref(res)
  endfunction()