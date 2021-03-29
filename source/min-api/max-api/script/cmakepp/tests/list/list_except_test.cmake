function(test)

  set(lstA a b c d e)
  set(lstB c e)

   list_except(lstA lstB)
   ans(res)
   assert(${res} EQUALS a b d)


   
endfunction()