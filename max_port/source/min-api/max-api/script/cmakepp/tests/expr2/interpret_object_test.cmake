function(test)


  
  set(exception "{'__$type__':'exception'}")

  define_test_function2(test_uut expr_parse "interpret_object" "")
  ## no tokens
  test_uut("${exception}") 
  ## too many tokens 
  test_uut("${exception}" a b c)
  ## invlaid token 
  test_uut("${exception}" a) 


  ## empty brace token
  test_uut("{expression_type:'object'}" "{}")
  ## ok - single value
  test_uut("{expression_type:'object'}" "{1}")
  ## ok - single key value
  test_uut("{expression_type:'object'}" "{a:1}")


  ## runtime tests
  define_test_function2(test_uut expr_eval "interpret_object" "")

  ## empty object
  test_uut("{}" "{}")
  ##  single key value
  test_uut("{a:1}" "{a:1}")
  ##  double key value
  test_uut("{a:1,b:2}" "{a:1,b:2}")
  ## triple key value
  test_uut("{a:1,b:2,c:3}" "{a:1,b:2,c:3}")
  ## mixed key values and values
  test_uut("{a:1,c:3}" "{a:1,b,2,c:3}")
  ans(res)
  address_get("${res}")
  ans(res)
  assert("${res}" EQUALS b 2)
  ## list
  test_uut("{a:[1,2]}" {a:[1,2]})
  ## nested objects
  test_uut("{a:{b:'c'}}" {a:{b:c}})
  ## nested list objects
  test_uut("{a:[{b:1},{c:2}]}" { a:[{b:1},   {c:2}]})


endfunction()