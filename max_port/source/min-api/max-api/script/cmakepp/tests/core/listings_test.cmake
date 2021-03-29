function(test)




  listing()
  ans(uut)
  listing_append(${uut} "call('hello;asd')")
  listing_compile(${uut})
  ans(res)
  assert("${res}" EQUALS "call(\"hello;asd\")")

  listing_begin()
    line("test()")

  listing_end_compile()
  ans(res)
  assert(${res} STREQUAL "test()")




  listing_begin()
    line("function(fun)")
    line("message('hello')")
    line("endfunction()")
  listing_end_compile()
  ans(res)
  assert("${res}" STREQUAL "function(fun)\n  message(\"hello\")\nendfunction()")


  define_test_function(test_uut listing_make_compile)


  test_uut("test(123)" "test(123)")
  test_uut("test(123)\ntest(123)" "test(123)" "test(123)")
  test_uut("macro()\n  message(asd)\nendmacro()" "macro()" "message(asd)" "endmacro()")
  test_uut("call(\${asd} \${bsd_csd} \${dsd_sdk})" "call({asd} {bsd_csd} {dsd_sdk})")
  test_uut("call(\"hello\")" "call('hello')")




  listing()
  ans(uut)
  listing_append(${uut} function(test_func))
  listing_append(${uut} return(123))
  listing_append(${uut} endfunction())
  listing_compile(${uut})
  ans(res)
  assert(${res} STREQUAL "function( test_func )\n  return( 123 )\nendfunction( )")


  listing()
  ans(uut)
  listing_append(${uut} "test({asd})")
  listing_compile(${uut})
  ans(res)
  assert(${res} STREQUAL "test(\${asd})")


endfunction()