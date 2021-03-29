function(test)


  set(outer_scope asd)
  timer_start(t1)
  eval_cmake(
    function(this_is_a_test varA
      varB
      varC)
      return("ok \${varA} \${varB} \${varC} \${ARGN} ${outer_scope} \${outer_scope}")
    endfunction()
    set_ans("nananana")
    )
  ans(res)
  set(outer_scope bsd)
  timer_print_elapsed(t1)

  assert("${res}" STREQUAL "nananana")
  this_is_a_test(a b c d)
  ans(res)
  assert("${res}" STREQUAL "ok a b c d asd bsd" )

endfunction()