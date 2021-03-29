function(test)


  define_test_function(test_uut command_line_parse)  




  ## test command
  test_uut("{command:'test'}" test)
  test_uut("{command:'test'}" COMMAND test )
  test_uut("{command:'test'}" COMMAND test a b c )
  test_uut("{command:'test'}" "test a b c" )
  test_uut("{command:'test a'}" "'test a' a b c" )
  test_uut("{command:'test a'}" "<test a> a b c" )
  test_uut("{command:'test a'}" COMMAND "test a" a b c) 
  


endfunction()