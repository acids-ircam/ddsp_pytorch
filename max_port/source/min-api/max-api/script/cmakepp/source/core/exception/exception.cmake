## `(<exception>|<any>)-> <exception>`
##
## transforms the input into an exception
## or returns the input exception
function(exception)
  is_exception("${ARGN}")
  ans(is_exception)
  if(is_exception)
    return("${ARGN}")
  endif()
  exception_new("${ARGN}")
  return_ans()
endfunction()