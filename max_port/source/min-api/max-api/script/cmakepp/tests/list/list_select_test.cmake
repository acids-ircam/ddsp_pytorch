function(test)


  ## indices [0,4)
  index_range(0 4)
  ans(res)

  ##square each value
  list_select(res "
    function(func i)
      return_math(\"\${i} * \${i}\")
    endfunction()")
  ans(squared)


  assert(${squared} EQUALS 0 1 4 9)



  ## test for issues # 68
  set(files filea fileb)
   assign(sources = list_select(files "function(_ el)\nreturn(\"\${el}.c\")\nendfunction()"))
   assign(headers = list_select(files "function(_ el)\nreturn(\"\${el}.h\")\nendfunction()"))

   assert(${sources} EQUALS filea.c fileb.c)
   assert(${headers} EQUALS filea.h fileb.h)

endfunction()