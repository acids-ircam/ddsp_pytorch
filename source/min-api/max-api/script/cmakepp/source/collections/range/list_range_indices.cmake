
  ## list_range_indices(<list&> <range ...>)
  ## returns the indices for the range for the specified list
  ## e.g. 
  ## 
  function(list_range_indices __lst)
    list(LENGTH ${__lst} len)
    range_indices("${len}" ${ARGN})
    ans(indices)
    return_ref(indices)
  endfunction()

