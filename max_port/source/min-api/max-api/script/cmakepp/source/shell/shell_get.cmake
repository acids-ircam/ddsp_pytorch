# returns which shell is used (bash,cmd) returns false if shell is unknown
function(shell_get)
  if(WIN32)
    return(cmd)
  else()
    return(bash)
  endif()
endfunction()
