##
##
##
function(is_range)
  if("_${ARGN}_" MATCHES "^_[0-9:\\$,\\-]*_$")
    return(true)
  endif()
  return(false)    
endfunction()