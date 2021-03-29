function(test)


  index_range(0 10)
  ans(range)



  list_all(range "
    function(func i)
      return(true)
    endfunction()
    ")
  ans(res)


  assert(res)


  list_all(range "
    function(func i)
      return(false)
    endfunction()
    ")
  ans(res)
  assert(NOT res)


  list_all(range "
    function(func i)
      if(\${i} LESS 20 )
        return(true)
      endif()
      return(false)
    endfunction()")
  ans(res)
  assert(res)


endfunction()