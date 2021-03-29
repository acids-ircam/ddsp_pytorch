## `(<promise> <continuation:<promise>>)-><promise>`
##
## adds the continuation to the specified promise. 
## when the promise is resolved it will schedule continuation to be executed
function(promise_then promise continuation)
  promise("${promise}")
  ans(promise)

  promise("${continuation}")
  ans(continuation)

  promise_is_resolved("${promise}")
  ans(is_resolved)

  if(is_resolved)
    continuation_resolve("${promise}" "${continuation}")
  else()
    map_append("${promise}" continuations "${continuation}")
  endif()

  return_ref(continuation)
endfunction()
