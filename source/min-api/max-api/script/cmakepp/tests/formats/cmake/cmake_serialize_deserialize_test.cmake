function(test)


  
  function(test_cmake_serialize_deserialize)
  
    message("serializing '${ARGN}'")
    message(PUSH)
    data(${ARGN})
    ans(data)

    timer_start(qm_serialize)
    qm_serialize(${data})
    ans(qmdata)
    timer_print_elapsed(qm_serialize)


    timer_start(qm_deserialize)
    qm_deserialize("${qmdata}")
    timer_print_elapsed(qm_deserialize)

    timer_start(cmake_serialize)
    cmake_serialize(${data})
    ans(res)
    timer_print_elapsed(cmake_serialize)
#
 #   _message("${res}")

    timer_start(cmake_deserialize)
    cmake_deserialize("${res}")
    ans(res)
    timer_print_elapsed(cmake_deserialize)


    message(POP "   ")

    return_ref(res)
  endfunction()

  define_test_function(test_uut test_cmake_serialize_deserialize)
  test_uut("a;b;c" a b c)
  test_uut("{}" "{}")
  test_uut("" "")
  test_uut("abc" "abc")
  test_uut("{a:null}" "{a:null}")
  test_uut("{a:'abc'}" "{a:'abc'}")
  test_uut("{a:'abc',b:'cde'}" "{a:'abc',b:'cde'}")
  test_uut("{a:{}}" "{a:{}}")
  test_uut("{a:{b:'asdasd'},c:'kk'}" "{a:{b:'asdasd'},c:'kk'}")
return()
  test_cmake_serialize_deserialize(a b c)
  test_cmake_serialize_deserialize()
  test_cmake_serialize_deserialize("{}")
  cmake_deserialize("${lala}")
    ans(res)
    address_get(${res})
    ans(res)
    json_print(${res})
  return()
  test_cmake_serialize_deserialize("{a:{b:'asdasd'},c:'kk'}")



    
  map_new()
  ans(m1)
  map_set(${m1} a ${m1})

  cmake_serialize("${m1}")
  ans(res)
  #message("${res}")
  cmake_deserialize("${res}")
  ans(res)
  map_tryget(${res} a)
  ans(res2)
  ## check if cycles work
  assert(${res} STREQUAL ${res2})
  test_cmake_serialize_deserialize("{a:{b:{b:{b:{b:{b:{b:{b:{b:{b:{b:{b:{b:true}}}}}}}}}}}},c:'kk'}")


  endfunction()