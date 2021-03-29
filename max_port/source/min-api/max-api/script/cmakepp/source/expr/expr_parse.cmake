## `(<expression type> <arguments:<any...>> <expression>)-><expr ast>`
##
##
## parsers and caches the expression. returns the AST for the specified
## expression.  See `ast_new`
function(expr_parse type arguments)
  
  set(argn "${ARGN}")
  arguments_expression_parse_cached("${type}" "${arguments}" "argn" 2 ${ARGC})
  return_ans()
endfunction()

