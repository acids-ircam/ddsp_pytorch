function(test)



  index_range(0 10)
  ans(res)

  ## counts all uneven elements
  list_count(res "
    function(func i)
      return_math(\"\${i} % 2\")
    endfunction()")

  ans(res)
  assert(${res} EQUAL 5)



endfunction()