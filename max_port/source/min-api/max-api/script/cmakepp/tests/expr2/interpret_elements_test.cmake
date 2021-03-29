function(test)

set(exception "{'__$type__':'exception'}")

  ##### compile time tests ######

  ##  interpret elements
  define_test_function2(test_uut expr_parse interpret_elements "comma;interpret_interpolation")


  ## invalid token
  test_uut("${exception}" "$")  
  ## no token 
  test_uut("")  
  ## single valid token 
  test_uut("{expression_type:'literal'}" a)  
  ## single multi token element
  test_uut("{ expression_type:'literals', value:'ab' }" a b)  
  ## multi single token elements
  test_uut("[{ expression_type:'literal', value:'a'},{expression_type:'literal',value:'b'}]" a , b)  
  ## multi single token elements
  test_uut("[{ expression_type:'literals', value:'ab'},{ expression_type:'literals', value:'cd'}]" a b, c d)  



endfunction()