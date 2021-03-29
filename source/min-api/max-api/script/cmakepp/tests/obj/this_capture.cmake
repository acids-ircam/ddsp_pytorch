function(test)




  function(MyClass a b c)
    this_capture(asd:a bsd:c csd:b)
  endfunction()

   new(MyClass 1 2 3)
   ans(uut)

   assertf({uut.asd} EQUALS 1)
   assertf({uut.csd} EQUALS 2)
   assertf({uut.bsd} EQUALS 3)



endfunction()