

  function(listing_combine)
    listing()
    ans(lst)
    foreach(listing ${ARGN})
      address_get(${listing})
      ans(current)
      address_append("${lst}" "${current}")
    endforeach()
    return(${lst})
  endfunction()