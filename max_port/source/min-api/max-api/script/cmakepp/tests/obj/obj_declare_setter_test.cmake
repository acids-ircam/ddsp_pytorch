function(test)


  function(TestClass)

    this_declare_setter(setter)
    function(${setter} obj key)
      return("the value of ${key} is ${ARGN}")
    endfunction()

  endfunction()



  new(TestClass)
  ans(uut)

  obj_set(${uut} asd gugugu)
  ans(res)
  assert("${res}" STREQUAL "the value of asd is gugugu")


endfunction()