## `(<promise> <promise>|<any>...)-><bool>` 
##
## if input is a promise then the promise will be resolved after the input
## promise is resolved
##
## resolves the specified promise
## returns true if promise could be resolved
## returns false if promise was already resolved
## returns <null> if resolution was a promise and promise will be resolved later
function(promise_resolve promise)
  promise("${promise}")
  ans(promise)

  promise_is_resolved("${promise}")
  ans(is_resolved)
  if(is_resolved)
    #message("tried to re-resolve promise ${promise}")
    return(false)
  endif()



  is_promise("${ARGN}")
  ans(is_promise)
  if(is_promise)
   # message("trying to resolve promise with promise")
    promise_then("${ARGN}" "${promise}")
    return()
  endif()

  ## todo
  ##is_error("${ARGN}")
  ##ans(is_error)


  map_tryget("${promise}" continuations)
  ans(continuations)
  map_set("${promise}" value "${ARGN}")
  map_set("${promise}" promise_state "resolved")
 # message("resolving promise ${promise} with '${ARGN}' ")

  foreach(continuation ${continuations})
    continuation_resolve("${promise}" "${continuation}")
  endforeach()

  return(true)
endfunction()