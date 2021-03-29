function(test)
  set(exception "{'__$type__':'exception'}")
  
  define_test_function2(test_uut expr_parse "interpret_key_value" "")

  ## no tokens
  test_uut("${exception}") 
  ## too few tokens
  test_uut("${exception}" a)
  ## no colon 
  test_uut("${exception}" a b c)
  ## no key 
  test_uut("${exception}" ":b")
  ## invalid  key 
  test_uut("${exception}" ",:b")
  ## no value 
  test_uut("${exception}" "a:")
  ## invalid value
  test_uut("${exception}" "a:,")
  ## valid
  test_uut("{expression_type:'key_value'}" "a:b")
endfunction()