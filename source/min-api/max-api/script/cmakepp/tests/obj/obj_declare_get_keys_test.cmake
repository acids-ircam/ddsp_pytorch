function(test)
  
  

  function(TestClass)
    this_declare_get_keys(thefunc)
    function(${thefunc} obj)
      return(a b c)
    endfunction()
  endfunction()

  new(TestClass)
  ans(uut)
  obj_keys(${uut})
  ans(keys)

  assert(${keys} EQUALS a b c)

endfunction()