function(test)
  # function(func a b c)
  #   return(1)
  # endfunction()


  # function(call2 _function_ish _paren_open)
  #   set(args ${ARGN})
  #   list_pop_back(args)
  #   ans(_paren_close)



  #   if(COMMAND "${_function_ish}")
  #     eval("${_function_ish}(\${args})")
  #     return_ans()
  #   endif()

  #   message(FATAL_ERROR "cannot handle function")
  # endfunction()

  # timer_start(timer)
  # foreach(i RANGE 0 1000)
  #   #func( 1 2 3)
  #   call(func(a b c))
  # endforeach()
  # timer_print_elapsed(timer)


  # return()
  # simple call void => void 
  function(fu)
  return()
  endfunction()

  call(fu())
  ans(res)
  assert(NOT res)


  # return value void => value
  function(fu1)
    return("myvalue")
  endfunction()

  call(fu1())
  ans(res)
  assert("${res}" STREQUAL "myvalue")

  # value => value
  function(fu2 arg)
    return("${arg}${arg}")
  endfunction()

  call(fu2(1))
  ans(res)
  assert("${res}" STREQUAL "11")

  # value , value => value
  function(fu3 arg1 arg2)
  return("${arg2}${arg1}")
  endfunction()

  call(fu3(12 34))
  ans(res)
  assert("${res}" STREQUAL "3412")

  # variable function name void => value
  function(myfu arg1 arg2)
  return("${arg2}${arg1}")
  endfunction()
  set(var myfu)

  call(${var}(12 34))
  ans(res)
  assert("${res}" STREQUAL "3412")


  # imported function
  call("function(fruhu arg)\nreturn(\${arg}\${arg})\nendfunction()"(ab))
  ans(res)
  assert("${res}" STREQUAL "abab")

  # lambda function
  call("[]()message(hello)"())

  # 
  call("[]()return(muuu)"())
  ans(res)
  assert("${res}" STREQUAL "muuu")

  # 
  set(myfuncyvar "[]()return(mewto)")
  call(myfuncyvar())
  ans(res)
  assert("${res}" STREQUAL "mewto")

  #
  function(fu5)
    return("mewthree")
  endfunction()
  set(myfuncyvar fu5)
  call(myfuncyvar())
  ans(res)
  assert("${res}" STREQUAL "mewthree")




  nav(my.test.object "[]()return(mewfour)")
  call(my.test.object())
  ans(res)
  assert("${res}" STREQUAL "mewfour")



function(f6)
  return("f6")
endfunction()
set(f6b f6)
call(f6b())
ans(res)
assert("${res}" STREQUAL f6b)


function(TestClass44)
this_set(muha asd)
  this_declare_call(op)
  function(${op})
    return("callop ${ARGN}")
  endfunction()
  proto_declarefunction(func)
  function(${func})
    this_get(muha)
    return("func ${muha} ${ARGN}")
  endfunction()
endfunction()

obj_new(TestClass44)
ans(obj2)



call(${obj2}(1))
ans(res)
assert("${res}" STREQUAL "callop 1")

call(obj2(1))
ans(res)
assert("${res}" STREQUAL "callop 1")

call(obj2.func(1))
ans(res)
assert("${res}" STREQUAL "func asd 1")


endfunction()