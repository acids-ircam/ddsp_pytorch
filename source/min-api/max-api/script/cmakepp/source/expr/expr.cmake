## `(<expression>)-><any>`
##
## parses, compiles and evaluates the specified expression. The compilation result
## is cached (per cmake run)
##
function(expr)
  set(argn "${ARGN}")
  arguments_expression_eval_cached("interpret_expression" "" argn 0 ${ARGC})
  rethrow()
endfunction()