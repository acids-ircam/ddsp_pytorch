
  function(regex_match regex)
    string(REGEX MATCH "${regex}" ans ${ARGN})
    return_ref(ans)
  endfunction()