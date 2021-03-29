function(test)

  set(exception "{'__$type__':'exception'}")


  ##### compile time tests ######

  define_test_function2(test_uut expr_parse interpret_literal "")

  ## 1 token required
  test_uut("${exception}") 
  ## only single token
  test_uut("${exception}" a b) 
  ## unquoted empty string
  test_uut("{expression_type:'literal', value_type:'unquoted_string', value:''}" "") 
  ## single sepearted token
  test_uut("{expression_type:'literal', value_type:'unquoted_string', value:'a'}" "a") 
  ## single separated token with space
  test_uut("{expression_type:'literal', value_type:'unquoted_string', value:'a\\\\ b'}" "a b") 
  ## single single quoted token 
  test_uut("{expression_type:'literal', value_type:'single_quoted_string', value:'a'}" "'a'") 
  ##  single double quoted token 
  test_uut("{expression_type:'literal', value_type:'double_quoted_string', value:'a'}" "\"a\"") 
  ## number
  test_uut("{expression_type:'literal', value_type:'number', value:'123'}" "123") 
  ## bool, true  
  test_uut("{expression_type:'literal', value_type:'bool', value:'true'}" "true") 
  ## bool, false
  test_uut("{expression_type:'literal', value_type:'bool', value:'false'}" "false") 
  ## null
  test_uut("{expression_type:'literal', value_type:'null', value:''}" "null") 




  ##### runtime tests #####

  #event_addhandler(on_exception "[](ex) print_vars(ex)")

  define_test_function2(test_uut expr_eval interpret_literal "")

  ## literal test
  test_uut("${exception}")
  test_uut("${exception}" a b) # => error only single token allowed
  test_uut("${exception}" "[") # => error  invalid token
  test_uut("${exception}" ,) # => error  invalid token
  test_uut("" "")
  test_uut("a" "a") 
  test_uut("" "''") 
  test_uut("" "\"\"") 
  test_uut("a b c" "a b c") ## sepearated arg
  test_uut("a" "'a'") 
  test_uut("a" "\"a\"") 
  test_uut("123" "123")
  test_uut("true" "true")
  test_uut("false" "false")
  test_uut("" "null")
  test_uut("null" "'null'")
  test_uut("abc def" "abc def")

endfunction()