
  function(semver_constraint_element_isvalid element)
    string(REGEX MATCH "^[~\\>\\<=!]?([0-9]+)(\\.[0-9]+)?(\\.[0-9]+)?(-[a-zA-Z0-9\\.-]*)?(\\+[a-zA-Z0-9\\.-]*)?$" match "${element}")
    if(match)
      return(true)
    else()
      return(false)
    endif()
  endfunction()