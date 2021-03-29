  ## compares the specified files
  ## returning true if their content is the same else false
  function(fequal lhs rhs)
    path_qualify(lhs)
    path_qualify(rhs)

    cmake(-E compare_files "${lhs}" "${rhs}" --exit-code)
    ans(error)
    
    if(error)
      return(false)
    endif()
    return(true)
  endfunction()
