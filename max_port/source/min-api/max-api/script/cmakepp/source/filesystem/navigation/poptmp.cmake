##
##
## removes the current path from the path stack
## delets the temporary directory
function(poptmp)
  pwd()
  ans(pwd)
  if(NOT "${pwd}" MATCHES "mktemp")
    message(FATAL_ERROR "cannot poptmp - path ${pwd} is not temporary ")
  endif() 
  rm(-r "${pwd}")
  popd()
  return_ans()
endfunction()