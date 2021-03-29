function(test)
 

  define_test_function2(test_uut expr_parse interpret_default_value "")



  define_test_function2(test_uut expr_eval interpret_default_value "")
  set(a)
  test_uut("{}" "$a?.b?")

  expr("$a?.b?.c?")
  assertf({a.b.c} ISNOTNULL)

endfunction()