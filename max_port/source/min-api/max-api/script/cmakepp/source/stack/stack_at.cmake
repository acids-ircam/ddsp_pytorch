
# returns the specified element of the stack
function(stack_at stack idx)
  map_tryget("${stack}" back)
  ans(current_index)
  math(EXPR idx "${idx} + 1")
  if("${current_index}" LESS "${idx}")
    return()
  endif()
  map_tryget("${stack}" "${idx}")
  return_ans()
endfunction()