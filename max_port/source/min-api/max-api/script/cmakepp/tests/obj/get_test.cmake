function(test)


  function(TestClass)
    this_declare_getter(getter)
    function(${getter} obj prop)
      return(${obj})
    endfunction()
  endfunction()


  new(TestClass)
  ans(uut)

  ## gets objects
  get(val = uut.val.asd.qwe.sad.zxc)
  assert("${val}" STREQUAL "${uut}")


  ## gets from a map
  obj("{a:{b:{c:1}}}")
  ans(uut)
  get(val = uut.a.b.c)
  assert("${val}" EQUAL "1")




endfunction()