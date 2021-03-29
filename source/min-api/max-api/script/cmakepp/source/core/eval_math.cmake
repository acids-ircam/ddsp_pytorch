# evaluates a cmake math expression and returns its
# value
function(eval_math)
  math(EXPR res ${ARGN})
  return_ref(res)
endfunction()
