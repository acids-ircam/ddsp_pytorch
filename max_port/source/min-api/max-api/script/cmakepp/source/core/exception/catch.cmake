## `(<exception_handler: <anonymous function>>)-><error>|<any>`
##
## catches an exception using the specified exception handler
## if no exception handler is specified the catch function returns nothing
## which is not an exception
## the exception handler may throw a new exception 
## which can be caught as well
function(catch)
  set(exception "${__ans}")
  is_exception("${exception}")
  ans(is_exception)
  
  if(NOT is_exception)
    _return()
  endif()

  address_pop_back(unhandled_exceptions)
  address_push_back(handled_exceptions ${exception})

  ## a single argument is interpreted as a variable
  if(${ARGC} EQUAL 1)
    ## maybe handle callable here as well?
    set(${ARGV0} ${exception} PARENT_SCOPE)
    return()
  endif()

  arguments_anonymous_function(0 ${ARGC})
  ans(exception_handler)
  
  if(NOT exception_handler)
    return()
  endif()

  eval("${exception_handler}(${exception})")
  return_ans()
endfunction()