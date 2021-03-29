
  # combines all dirs to a single path  
  function(path_combine )
    set(args ${ARGN})
    list_to_string(args "/")
    ans(path)
    return_ref(path)
  endfunction()