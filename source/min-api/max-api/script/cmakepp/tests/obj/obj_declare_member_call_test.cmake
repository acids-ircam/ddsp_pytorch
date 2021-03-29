function(test)
  
  function(TestClass)

    this_declare_member_call(membercall)
    function(${membercall} obj membername)
      return("${membername}(${ARGN})")
    endfunction()

  endfunction()

  new(TestClass)
  ans(uut)


  rcall(res = uut.asdasd(1 2 3))
  
  assert("${res}" EQUALS "asdasd(1;2;3)")



endfunction()