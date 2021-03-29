function(test)
  set(exception "{'__$type__':'exception'}")

  ##### runtime tests #####

  ## scope rvalue

  define_test_function2(test_uut expr_eval interpret_scope_rvalue "")

  set(the_var 123)

  set(the_other_var)

  test_uut("" "$[the_other_var]") ## ok - no value

  test_uut("${exception}") ## no tokens
  test_uut("${exception}" "the_other_var") ## no dollar symbol
  test_uut("${exception}" "$") ## no identifier or paren
  test_uut("" "$(the_other_var)") ## ok - no value
  test_uut("" "$the_other_var") ## ok - no value
  test_uut("" "$'the_other_var'") ## ok - no value
  test_uut("" "$\"the_other_var\"") ## ok - no value
  test_uut("123" "$the_var")
  test_uut("123" "$[the_var]") ## ok
  test_uut("123" "$(the_var)") ## ok
   # test_uut("123" "$.the_var") ## ok should be ok

endfunction()