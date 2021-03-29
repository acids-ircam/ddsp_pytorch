function(test)


  script("{a:{b:{c:3}}}")
  ans(obj)

  timer_start(t1)
  foreach(i RANGE 1 1000)
    ref_nav_get("${obj}" a b c)
  endforeach()
  timer_print_elapsed(t1)

  timer_start(t2)
  foreach(i RANGE 1 1000)
    map_navigate(res "obj.a.b.c")
  endforeach()
  timer_print_elapsed(t2)


#return()

  script("{a:{b:{c:3}}}")
  ans(obj)
  assign(res = obj.a.b.c.d)
  assert(NOT res)

  assign(res = obj.a.b.c)
  assert("${res}" STREQUAL "3")

  assign(res = obj.a.b.c.d)
  assert(NOT res)

  assign(res = obj.a.b.c)
  assert("${res}" STREQUAL "3")

  ## some data structures for testing

  function(some_class)
    this_set(asd 123)
    method(mymethod)
    function(${mymethod})
      this_get(asd)
      return(${asd} 321 ${ARGN})
    endfunction()
  endfunction()
  new(some_class)
  ans(uut1)

  function(testfunc2)
    return(234)
  endfunction()

  map_new()
  ans(uut2)
  map_set(${uut2} func testfunc2)
  map_set(${uut2} func2 a testfunc2 c)
  map_set(${uut2} func3 testfunc2 testfunc2 testfunc2)



  function(some_other_type)
    this_declare_call(call)
    function(${call})
      return(${ARGN})
    endfunction()
  endfunction()
  new(some_other_type)
  ans(uut3)


  map_new()
  ans(uut4)
  ### assertions:


  # concats
  set(res asd)
  assign(res += '123')
  ans(res2)
  assert(${res} STREQUAL "asd123")
  assert(${res2} STREQUAL "asd123")

  return()

  # assigns

  set(res)
  assign(res[] = '123')
  assert(${res} EQUALS 123)
  assign(res[] = '234')
  assert(${res} EQUALS 123 234)
  assign(res[0[ = '345')
  assert(${res} EQUALS 345 123 234)


  assign(uut4.a[] = '123')
  assign(uut4.a[] = '234')
  assign(uut4.a[0[ = '345')
  assertf({uut4.a} EQUALS 345 123 234)

  assign(res = '123')
  ans(res)
  assert(${res} EQUALS 123)

  assign(res = "{a:1}")
  assert(${res} MAP_MATCHES "{a:1}")


  assign(res.a = '{a:23}')
  assert(${res} MAP_MATCHES "{a:23}")

  set(res)
  data("{a:{b:[1,2,3,4,5,{c:6}]}}")
  ans(data)
  assign(!res.a.b.c.d.e = data.a.b[5].c)
  assertf({res.a.b.c.d.e} EQUAL 6) 

  set(input 123)
  assign(res = input)
  assert(${res} EQUALS 123)



  ## calls
  assign(res = testfunc2())
  assert(${res} EQUALS 234)

  assign(res = uut1.mymethod(876))
  assert(${res} EQUALS 123 321 876)

  assign(res = "[]()return(100)"())
  assert(${res} EQUALS 100)

  assign(res = uut2.func())
  assert(${res} EQUALS 234)

  assign(res = uut2.func2[1]())
  assert(${res} EQUALS 234) 

  assign(res = uut2.func3[:]())
  assert(${res} EQUALS 234 234 234)



  assign(res = uut3(hello))
  assert(${res} EQUALS hello)

  function(identity)
    return(${ARGN})
  endfunction()

  assign(res = identity("{a:123}"))
  assertf({res.a} EQUALS 123)


endfunction()