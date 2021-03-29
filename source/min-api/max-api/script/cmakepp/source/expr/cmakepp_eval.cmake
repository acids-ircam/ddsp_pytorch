## `(<cmakepp code>)-><any>`
##
## evaluates the specified cmakepp code
function(cmakepp_eval input)
  cmakepp_expr_compile("${input}")
  ans(____code____)
  eval_ref(____code____)
  return_ans()
endfunction()
