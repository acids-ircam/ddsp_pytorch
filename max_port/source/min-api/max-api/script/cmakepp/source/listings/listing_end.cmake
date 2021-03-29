
  function(listing_end)
    set(lst ${__listing_current})
    set(__listing_current PARENT_SCOPE)
    return_ref(lst)
  endfunction()