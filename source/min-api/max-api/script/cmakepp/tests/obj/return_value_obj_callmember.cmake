function(test)
  
  function(TheTestClass)

    proto_declarefunction(method)
    function(${method})

      return(hello)
    endfunction()
  endfunction()


  obj_new( TheTestClass)
  ans(uut)
  obj_member_call(${uut} method)
  ans(res)

  assert(${res} STREQUAL "hello")

endfunction()