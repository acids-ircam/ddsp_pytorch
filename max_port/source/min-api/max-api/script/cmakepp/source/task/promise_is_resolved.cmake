## `(<promise>)-><bool>`
##
## returns true if the promise is resolved
function(promise_is_resolved promise)
  promise_state("${promise}")
  ans(state)
  if("${state}" STREQUAL "resolved")
    return(true)
  endif()
  return(false)
endfunction()

