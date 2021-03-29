function(test)
  set(exception "{'__$type__':'exception'}")

  ##### runtime tests #####


  ## navigation rvalue
  obj("{
    b:1,
    c:{
      d:2,
      e:[
        {f:3},
        {g:4},
        5,
        6,
        {h:7}
      ]
    },
    i:[8,9,10],
    j:{},
    k:[]
  }")
  ans(a)
  
  define_test_function2(test_uut expr_eval interpret_navigation_rvalue "")
  
  test_uut("1" "$a.b") 
  test_uut("${exception}")
  test_uut("${exception}" "a") # too few tokens
  test_uut("${exception}" "abc" "abc") # missing dot
  test_uut("${exception}" ".abc") # no lvalue 
  test_uut("" "a.abc") # ok 
  test_uut("2" "$a.c.d")
  test_uut("8;9;10" "$a.i")






endfunction()