function(test)





return()






return()
  ## interpret assign

function(alal)
  map_new()
  ans(asd)
  map_set(${asd} lol 123)
  return(${asd})
endfunction()
  obj("{b:{c:2}}")
  ans(a)
  expr(interpret_assign "--print-code" "$a.b.c = alal().lol")
  ans(res)
  assert("${res}" STREQUAL 123)
  assertf("{a.b.c}" STREQUAL 123)

return()
  set(a)
  expr(interpret_assign "" "$a = 1")
  ans(res)
  assert("${res}" EQUAL 1)
  assert("${a}" EQUAL 1)


return()


  ## intepret statements
  define_test_function2(test_uut expr interpret_statements "")

  expr(interpret_statements "" "a;b")
  ans(res)
  assert("${res}" STREQUAL b)


  expr(interpret_statements "" "a;b;c")
  ans(res)
  assert("${res}" STREQUAL c)

  test_uut("${exception}")
  test_uut("" "")
  test_uut("a" "a")











  return()





  function(interpret_range tokens)
    list_select_property(tokens type)
    ans(token_types)
    list_remove(token_types number dollar colon minus comma)
    if(token_types)
      throw("unexpected token types" --function interpret_range)
    endif()




    ast_new(
      "${tokens}"         # tokens
      "range"             # expression_type
      ""                  # value_type
      ""                  # ref
      ""                  # code
      ""                  # value
      ""                  # const
      ""                  # pure_value
      ""                  # children
      )
    ans(ast)


    return()
    throw("not implemented")

  endfunction()

  define_test_function2(test_uut expr "interpret_range" "--ast")


  ## invlaid token
  test_uut("${exception}" abc)

#  test_uut("${exception}")


  return()




endfunction()