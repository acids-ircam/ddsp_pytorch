function(test)


  macro(arguments_encoded_list_test)
    string_codes()
    set(str)
    foreach(i RANGE 0 100)
      set(str "${str}${free_token}\${ARGV${i}}")
    endforeach()
    string(SUBSTRING "${str}" 1 -1 str)
    set(eval_this "

      macro(arguments_encoded_list_test)
        string_encode_list(\"${str}\")
        string(REPLACE \"${free_token}${free_token}\" \"\" __ans \"\${__ans}\")
        string(REPLACE \"${free_token}\" \"\\;\" __ans \"\${__ans}\")

      endmacro()
    ")
   # _message("${eval_this}")
    eval("${eval_this}")

    arguments_encoded_list_test()

  endmacro()


  function(test)
    arguments_encoded_list_test()
    return_ans()
  endfunction()

  functioN(test2)
    test(a b "c;d")
    return_ans()
  endfunction()

  test2(a b c d e f g h )
  ans(res)
  _message("${res}")
  test(a b c "a;b" "c;d")
  ans(res)
  _message("res '${res}'")


  timer_start(t1)
  foreach(i RANGE 0 1000)
    test(a b "c;d" e f)
  endforeach()
  timer_print_elapsed(t1)
  timer_start(t1)
  foreach(i RANGE 0 1000)
    invocation_argument_encoded_list(a b "c;d" e f)
  endforeach()
  timer_print_elapsed(t1)

  invocation_argument_string()
  ans(res)
  assert("${res}_" STREQUAL "_")

  invocation_argument_string("a")
  ans(res)
  assert("${res}_" STREQUAL "a_")

  invocation_argument_string("a;b;c" d)
  ans(res)
  assert("${res}" EQUALS "\"a;b;c\" d")

  invocation_argument_string("")
  ans(res)
  assert("${res}" STREQUAL "\"\"")


  invocation_argument_string("" "" a "" )
  ans(res)
  assert("${res}" STREQUAL "\"\" \"\" a \"\"")

  invocation_argument_string(asdasd("1;2;3" a b c))
  ans(res)
  assert("[${res}]" STREQUAL "[asdasd ( \"1;2;3\" a b c )]")

  invocation_argument_encoded_list(adasda("1;2;3" a b c))
  ans(res)


endfunction()