function(test)


  # arrange
  fwrite("a.txt" "abc")
  fwrite("b.txt" "cde") 
  fwrite("c.txt" "cde") 



  fequal("a.txt" "b.txt")
  ans(res)
  assert(NOT res)


  fequal("a.txt" "a.txt")
  ans(res)
  assert(res)

  fequal("b.txt" "c.txt")
  ans(res)
  assert(res)

endfunction()