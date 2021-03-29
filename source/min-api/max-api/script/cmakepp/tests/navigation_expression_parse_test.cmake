function(test)

  navigation_expression_parse("a.b.c")
  ans(res)
  assert(${res} EQUALS a b c)

  navigation_expression_parse("a.b[3].c")
  ans(res)
  assert(${res} EQUALS a b <3> c)

  navigation_expression_parse("a.b[3][34].c")
  ans(res)
  assert(${res} EQUALS a b <3> <34> c)

  navigation_expression_parse("[1]")
  ans(res)
  assert(${res} EQUALS <1>)

  navigation_expression_parse("[]")
  ans(res)
  assert(${res} EQUALS <>)

  navigation_expression_parse("[1][0]")
  ans(res)
  assert(${res} EQUALS <1> <0>)

  navigation_expression_parse("[1]" "[0]")
  ans(res)
  assert(${res} EQUALS <1> <0>)


endfunction()