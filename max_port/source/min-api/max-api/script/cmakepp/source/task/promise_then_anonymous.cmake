## `(<promise> <inline cmake code>)-><promise>`
##
## adds an inline continuation to the specified promise
function(promise_then_anonymous promise)
  arguments_anonymous_function(1 ${ARGC})
  ans(function)
  promise_from_callable("${function}")
  ans(continuation)
  promise_then("${promise}" "${continuation}")
  return_ans()
endfunction()

