function(test)



  index_range(0 5)
  ans(range)

  list_any(range "
    function(func i)
      if(\${i} EQUAL 3)
        return(true)
      endif()
      return(false)
    endfunction()
    ")

  ans(res)
  assert(res)




  list_any(range "function(func i)
      if(\${i} EQUAL 98)
        return(true)
      endif()
      return(false)
    endfunction()
    ")

  ans(res)
  assert(NOT res)



endfunction()