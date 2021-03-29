function(test)


  
  set(exception "{'__$type__':'exception'}")



  ##### compile time tests ######
  define_test_function2(test_uut expr_parse interpret_interpolation "")
  ## no token
  test_uut("${exception}") 
  ## single unquoted empty string
  test_uut("{expression_type:'literal', value_type:'unquoted_string', value:''}" "") 
   ## single unquoted string
  test_uut("{expression_type:'literal', value_type:'unquoted_string', value:'a'}" a) 
  ## two unquoted strings are concatenated
  test_uut("{expression_type:'literals', value_type:'composite_string', value:'ab'}" a b) 
  ## two nummbers are concatenated
  test_uut("{expression_type:'literals', value_type:'composite_string', value:'12'}" 1 2) 


  ##### runtime tests #####

  define_test_function2(test_uut expr_eval interpret_interpolation "")

  ## literals test

  # no token => exception
  test_uut("${exception}") 
  test_uut("abc" a b c)
  test_uut("abc d" 'a' \"b\" "c d")
  test_uut("${exception}" 'a' ( \"b\" "c d")) #invalid tokens

endfunction()