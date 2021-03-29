function(test)

  function(test_0)
    arguments_expression(0 ${ARGC})
    ans(rest)
    assert(${rest} ISNULL )
  endfunction()
  test_0()


  function(test_1)
    arguments_expression(0 ${ARGC})
    ans(rest)
    encoded_list_decode("${rest}")
    ans(rest)
    assert("${rest}" EQUALS "a;b;c")
  endfunction()

  test_1([a,b,c])


  ## checks wether spreading arguments correclty works
  function(test_2)
    arguments_expression(0 ${ARGC} v1 v2)
    ans(rest)
    assert("${v1}" EQUALS "a;b;c")
    assert("${v2}" EQUALS "d")
    assert("${rest}" EQUALS "e;f;g")
  endfunction()
  test_2([a,b,c], [d,e]..., f,g)


  ## test wether named args are correctly added
  function(test_3)
    arguments_expression(0 ${ARGC} v1 v2)
    assert(${v1} EQUALS a)
    assert(${v2} EQUALS b c)
  endfunction()
  test_3(v1:a, v2:[b,c])


  ## test wehter unused named args are igored
  function(test_4)
    set(v1 asd)
    arguments_expression(0 ${ARGC} v2)
    assert(${v1} EQUALS asd)
    assert(${v2} EQUALS csd)
    assertf({arguments_expression_result.v1} EQUALS bsd)
  endfunction()
  test_4(v1:bsd, v2:csd)

  ## test wether positional args are extracted correclty
  function(test_5)
    arguments_expression(0 ${ARGC} v1 v2 v3)
    ans(rest)
    assert(${v1} EQUALS bsd)
    assert(${v2} EQUALS asd)
    assert(${v3} EQUALS csd)
    assert(${rest} EQUALS dsd)

  endfunction()
  test_5(asd, v1: bsd, v3: csd, dsd)










endfunction()