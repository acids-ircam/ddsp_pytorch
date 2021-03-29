function(test)




  set(mylist 1 2 3 4 5)
  function(foldr a b)
    return_math("${a} + ${b}")
  endfunction()
  list_fold(mylist "foldr")
  ans(res)
  assert("${res}" EQUAL 15)


  list_fold(mylist "[](a b)return_math('{{a}} + {{b}}')")
  ans(res)
  assert("${res}" EQUAL 15)




  index_range(1 1000)
  ans(range)


  timer_start(timer)
  list_fold(range "[](a b)return_math('{{a}} + {{b}}')")
  ans(res)
  timer_print_elapsed(timer)

  message("res ${res}")


endfunction()