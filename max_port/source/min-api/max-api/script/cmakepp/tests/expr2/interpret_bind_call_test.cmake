function(test)

  set(exception "{'__$type__':'exception'}")
  ##### runtime tests #####


  ## compile time test
  define_test_function2(test_uut expr_parse "interpret_bind_call" "--ast")

  ## no tokens
  test_uut("${exception}") 
  ## no parens
  test_uut("${exception}" abc)
  test_uut("${exception}" abc bcd ccd dcd)
  ## no double colon
  test_uut("${exception}" "abc()")
  test_uut("${exception}" "abc:abc()")

  ## no lhs rvalue 
  test_uut("${exception}" "::abc()")
  ## invalid lhs rvalue 
  test_uut("${exception}" ",::abc()")
  ## invalid rhs rvalue
  test_uut("${exception}" "abc::,()")


  # valid
  test_uut("{expression_type:'bind_call'}" "abc::abc()")
  ## no rvalue

  
  ## valid  

  ## run time test
  define_test_function2(test_uut expr_eval "interpret_bind_call" "")
  test_uut(3 "abc::string_length()")



endfunction()