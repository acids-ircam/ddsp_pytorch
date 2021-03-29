
  function(regex_replace regex replace)
    string(REGEX REPLACE "${regex}" "${replace}" ans ${ARGN})
    return_ref(ans)
  endfunction()