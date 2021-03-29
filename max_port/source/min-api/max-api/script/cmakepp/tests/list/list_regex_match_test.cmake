function(test)





  set(lstA a b c d e f g)
  set(lstB)


  list_regex_match(lstA "[acfh]")
  ans(res)
  assert(${res} EQUALS a c f)

  list_regex_match(lstB "[abcdef]")
  ans(res)
  assert(${res} ISNULL)

  list_regex_match(lstA "[a]" "[e-g]")
  ans(res)
  assert(${res} EQUALS a e f g)




  # index_range(1 10000)
  # ans(range)
  # timer_start(timer)
  # list_regex_match(range "1$")
  # ans(res)
  # timer_print_elapsed(timer)


endfunction()