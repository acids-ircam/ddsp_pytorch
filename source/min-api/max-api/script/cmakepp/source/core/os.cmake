# returns the identifier for the os being used
function(os)
  if(WIN32)
    return(Windows)
  elseif(UNIX)
    return(Linux)
  else()
    return()
  endif()


endfunction()
