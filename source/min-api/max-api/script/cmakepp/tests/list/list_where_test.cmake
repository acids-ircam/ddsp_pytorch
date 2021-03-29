function(test)
  index_range(0 8)
  ans(res)

  ## select those elements which are uneven
  list_where(res "
    function(i i)
      return_math(\"\${i} % 2\")
    endfunction()")
  ans(res)
  assert(${res} EQUALS 1 3 5 7)

endfunction()