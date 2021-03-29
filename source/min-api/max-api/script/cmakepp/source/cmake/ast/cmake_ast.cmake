## `(<ast: <cmake code> |<cmake ast>>)-><cmake ast>`
##
## tries to parse the cmake code to an ast or returns the existign ast
function(cmake_ast ast)
  is_address("${ast}")
  ans(isref)
  if(NOT isref)
    cmake_ast_parse("${ast}")
    ans(ast)
  endif()
  return_ref(ast)
endfunction()

