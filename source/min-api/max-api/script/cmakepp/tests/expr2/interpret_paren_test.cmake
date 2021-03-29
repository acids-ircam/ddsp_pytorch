function(test)

 set(exception "{'__$type__':'exception'}")

  ##### compile time tests ######




  define_test_function2(test_uut expr_parse interpret_paren "")


  ## too few tokens
  test_uut("${exception}")
  ## too many tokens
  test_uut("${exception}" 1 2)
  ## right amount tokens but wrong type
  test_uut("${exception}" 1)
  ## right token empty paren is invalid 
  test_uut("${exception}" "()")
  ## right token empty expression in paren is invalid 
  test_uut("${exception}" "(,)")
  ## ok literal inner expression
  test_uut("{expression_type:'paren', children:{value:'a'}}" "(a)")

endfunction()