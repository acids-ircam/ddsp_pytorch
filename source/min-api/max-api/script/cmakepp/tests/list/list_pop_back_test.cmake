function(test)



  index_range(1 1000)
  ans(lst)
  timer_start(timer)
  while(true)
    list_pop_back(lst)
    ans(res)
    if(NOT res)
      break()
    endif()

  endwhile()
  timer_print_elapsed(timer)


  set(lst 1 2 3 4 )
  list_pop_back( lst)
  ans(res)
  assert(EQUALS 1 2 3 ${lst})
  assert("${res}" STREQUAL 4)


  ## tests if the correct result is returned for special values
  set(lst ";0;NOTFOUND; ;asd;bsd; ; ;")
  

  list_pop_back( lst)
  ans(arg)
  assert("${arg}_" STREQUAL "_")

  list_pop_back( lst)
  ans(arg)
  assert("${arg}" STREQUAL " ")

  list_pop_back(lst)
  ans(arg)
  assert("${arg}" STREQUAL " ")
  
  list_pop_back( lst)
  ans(arg)
  assert("${arg}" STREQUAL "bsd")
  
  list_pop_back( lst)
  ans(arg)
  assert("${arg}" STREQUAL "asd")
  
  list_pop_back( lst)
  ans(arg)
  assert("${arg}" STREQUAL " ")
  
  list_pop_back( lst)
  ans(arg)
  assert("${arg}" STREQUAL "NOTFOUND")
  
  list_pop_back( lst)
  ans(arg)
  assert("${arg}" STREQUAL "0")
  
  list_pop_back( lst)
  ans(arg)
  assert("${arg}_" STREQUAL "_")




  


endfunction()