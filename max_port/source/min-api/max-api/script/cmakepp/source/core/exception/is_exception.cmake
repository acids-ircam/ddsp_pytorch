
## `(<any>...)-><bool>`
##
## returns true iff the specified value is an exception
function(is_exception)
  map_get_special("${ARGN}" $type)
  ans(type)
  if("_${type}" STREQUAL "_exception")
    return(true)
  endif()
  return(false)
endfunction()
