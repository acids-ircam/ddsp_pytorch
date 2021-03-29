
  function(cmake_ast_serialize ast)
    cmake_ast("${ast}")
    ans(ast)
    assign(start = ast.tokens[0])
    assign(end = ast.tokens[$])
    cmake_token_range_serialize("${start};${end}")
    return_ans()
  endfunction()
