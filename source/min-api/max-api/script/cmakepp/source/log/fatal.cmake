## reports an error and stops program exection 
function(fatal)
  log(--error ${ARGN})
  ans(entry)

  map_tryget("${entry}" message)
  ans(message)

  _message(FATAL_ERROR "aborting exectution because of error '${message}'")
  return()
    
endfunction()
