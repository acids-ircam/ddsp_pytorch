function(test)



  set(var Tobias)
  function(func)
    return_ref(var)
  endfunction()

  function_capture(func var as uut)
  ans(func)

  set(var Becker)

  uut()
  ans(res)
  assert(res STREQUAL "Tobias")

  func()
  ans(res)
  assert(res STREQUAL "Becker")
  
endfunction()