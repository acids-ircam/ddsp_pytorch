function(test)

  set(exception "{'__$type__':'exception'}")

  define_test_function2(test_uut expr_parse interpret_navigation_lvalue "")


  ## no token
  test_uut("${exception}" "")
  ## to few tokens
  test_uut("${exception}" "a")
  



endfunction()