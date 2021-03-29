
 function(semver_component_compare left right)
 # message("comapring '${left}' to '${right}'")
    string_isempty( "${left}")
    ans(left_empty)
    string_isempty( "${right}")
    ans(right_empty)

    # filled has precedence before nonempty
    if(left_empty AND right_empty)
      return(0)
    elseif(left_empty AND NOT right_empty)
      return(1)
    elseif(right_empty AND NOT left_empty)
      return(-1)
    endif() 


    string_isnumeric( "${left}")
    ans(left_numeric)
    string_isnumeric( "${right}")
    ans(right_numeric)

    # if numeric has precedence before alphanumeric
    if(right_numeric AND NOT left_numeric)
      return(-1)
    elseif(left_numeric AND NOT right_numeric)
      return(1)
    endif()


   
    if(left_numeric AND right_numeric)
      if(${left} LESS ${right})
        return(1)
      elseif(${left} GREATER ${right})
        return(-1)
      endif()
      return(0)
    endif()

    if("${left}" STRLESS "${right}")
      return(1)
    elseif("${left}" STRGREATER "${right}")
      return(-1)
    endif()

    return(0)
 endfunction()