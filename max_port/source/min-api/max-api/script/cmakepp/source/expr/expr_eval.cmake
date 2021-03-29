## `(<expression type> <arguments:<any>...> <expression>)-><any> 
##
## evaluets the specified expression using as the type of expression 
## specified.  also passes allong arguments to the parser
function(expr_eval type arguments)
  set(argn "${ARGN}")
  arguments_expression_eval_cached("${type}" "${arguments}" argn 2 ${ARGC})
  
  rethrow()

endfunction()
