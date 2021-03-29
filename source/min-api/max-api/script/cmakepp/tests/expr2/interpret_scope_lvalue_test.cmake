function(test)

  set(exception "{'__$type__':'exception'}")



  expr_parse(interpret_literal "" "a")
  ans(rvalue)

  define_test_function2(test_uut expr_parse "interpret_scope_lvalue" "")

  ## no tokens
  test_uut("${exception}")
  ## not dollar token
  test_uut("${exception}" a)
  ## no identifier token 
  test_uut("${exception}" $)
  ## invalid identifier token
  test_uut("${exception}" "$,")
  

  ## valid
  test_uut("{expression_type:'scope_lvalue'}" "$b")
  
endfunction()