function(test)
 set(exception "{'__$type__':'exception'}")

  ##### compile time tests ######


  define_test_function2(test_uut expr_parse interpret_list "")

  # invalid too few tokens
  test_uut("${exception}")
  # invalid too many tokens
  test_uut("${exception}" 1 2) 
  # invalid right amout of tokens but wrong type
  test_uut("${exception}" 1)
  # empty list
  test_uut("{expression_type:'list',children:''}" "[]")
  # single element list
  test_uut("{expression_type:'list',children:[{value:'a'}]}" "[a]")
  # multi element list
  test_uut("{expression_type:'list',children:[{value:'a'},{value:'b'}]}" "[a,b]")


  

endfunction()