function(test)


  function(test_ref_nav_set current_value expression)
    data("${ARGN}")
    ans(data)
    data("${current_value}")
    ans(current_value)
    ref_nav_set("${current_value}" "${expression}" ${data})
    return_ans()
  endfunction()

  define_test_function(test_uut test_ref_nav_set current_value expression)
  
  test_uut("123;234;345" "123;345" "[1[" 234)
  test_uut(123 "" "" 123)
  test_uut("{a:123}" "{}" "a" 123)
  test_uut("{a:123}" "{a:234}" "a" 123)
  test_uut(123 "" "[]" 123) 
  test_uut("234;123" "234" "[]" 123) 
  test_uut("{a:[234,123]}" "{a:234}" "a[]" 123)
  test_uut("{a:123}" "" "!a" 123)
  test_uut("{a:234}" "{a:123}" "!a" 234)
  test_uut("{a:{b:123}}" "" "!a.b" 123)
  test_uut("{a:{b:123},c:123}" "{c:123}" "!a.b" 123)
  test_uut("{a:{b:123},c:123}" "{a:123,c:123}" "!a.b" 123)
  test_uut("{a:[123,234,345]}" "{a:[123,345]}" "a[1[" 234)
  test_uut("{a:[123,{b:123},345]}" "{a:[123,345]}" "!a[1[.b" 123)
  test_uut("{a:{b:{c:123}}}" "{a:{b:{c:234}}}" "!a.b.c" 123)


  ## todo.  let ref_nav set range values ie ![:].a = 1232

endfunction()