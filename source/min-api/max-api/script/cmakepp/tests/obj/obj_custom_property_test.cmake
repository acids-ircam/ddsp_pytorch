function(test)




  function(MyClass)
    this_set(input ${ARGN})

    obj_declare_property_getter("${this}" "testprop" mygetter)
    function(${mygetter} obj key)
      return(${this} ${obj} ${key} ${ARGN})
    endfunction()

    obj_declare_property_setter("${this}" testprop mysetter)
    function(${mysetter} obj key)
      return(${this} ${obj} ${key} ${ARGN})
    endfunction()


    obj_declare_property(${this} prop2)
    function(${get_prop2})
      list(REVERSE ARGN)
      return(${ARGN})
    endfunction()
    function(${set_prop2})
      list(REVERSE ARGN)
      return(${ARGN})
    endfunction()

    obj_declare_property(${this} prop3 getp3)
    function(${getp3})
      list(REVERSE ARGN)
      return(${ARGN})
    endfunction()

    obj_declare_property(${this} prop4 --setter setp4)
    function(${setp4})
      list(REVERSE ARGN)
      return(${ARGN})
    endfunction()

    obj_declare_property(${this} prop5 --setter setp5 --getter getp5)
    function(${setp5}) 
      list(REVERSE ARGN)
      return(${ARGN})
    endfunction()

    function(${getp5})
      list(REVERSE ARGN)
      return(${ARGN})
    endfunction()



    property(prop6)
    function(${get_prop6})
      return(${ARGN})
    endfunction()
    function(${set_prop6})
      return(${ARGN})
    endfunction()
  endfunction()


  new(MyClass 123)
  ans(uut)

  obj_get(${uut} prop2)
  ans(res)
  assert(${res} EQUALS prop2 ${uut})

  obj_set(${uut} prop2 123)
  ans(res)
  assert(${res} EQUALS 123 prop2 ${uut})

  obj_get(${uut} prop3)
  ans(res)
  assert(${res} EQUALS prop3 ${uut})

  obj_set(${uut} prop4 123)
  ans(res)
  assert(${res} EQUALS 123 prop4 ${uut})

  obj_get(${uut} prop4)
  ans(res)
  assert(${res} ISNULL)


  obj_get(${uut} prop5)
  ans(res)
  assert(${res} EQUALS prop5 ${uut})

  obj_set(${uut} prop5 123)
  ans(res)
  assert(${res} EQUALS 123 prop5 ${uut})


  obj_get(${uut} testprop)
  ans(res)
  assert(${res} EQUALS ${uut} ${uut} testprop)

  obj_set(${uut} testprop 123)
  ans(res)
  assert(${res} EQUALS ${uut} ${uut} testprop 123)

  obj_set(${uut} prop6 123)
  ans(res)
  assert(${res} EQUALS ${uut} prop6 123)

  obj_get(${uut} prop6)
  ans(res)
  assert(${res} EQUALS ${uut} prop6) 

endfunction()