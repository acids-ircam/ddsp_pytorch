
  function(regex_all regex)
    string(REGEX MATCHALL "${regex}" ans ${ARGN})
    return_ref(ans)
  endfunction()