function(test)

  function(my_comp a b )
    if(${a} GREATER ${b})
      return(${a})
    endif()
    return(${b})
  endfunction()
  set(mylist 2 9 4 7 1 8 2)
  list_max(mylist my_comp)
  ans(res)
  assert(${res} EQUAL 9)

endfunction()