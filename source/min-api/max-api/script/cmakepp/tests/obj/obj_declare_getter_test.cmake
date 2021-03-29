function(test)

  function(TestClass)
    this_declare_getter(custom_getter)
    function(${custom_getter} obj name)
      return("value of ${name}")
    endfunction()
  endfunction()

  new(TestClass)
  ans(uut)


  obj_get(${uut} someprop)
  ans(res)
  assert(${res} STREQUAL "value of someprop")

  obj_get(${uut} someprop2)
  ans(res)
  assert(${res} STREQUAL "value of someprop2")




endfunction()