## `(<exception> | <any>)->`
##
## may be used in functions.  causes the function to 
## return with an exception which can be caught
macro(throw)
  exception("${ARGN}")
  ans(__exc)
  address_push_back(unhandled_exceptions "${__exc}")
  address_push_back(exceptions "${__exc}")
  event_emit(on_exception ${__exc})
  return_ref(__exc)
endmacro()