function(test)



  range_indices("0" :)
  ans(res)
  assert(${res} ISNULL)


  range_indices("0" "0:n")
  ans(res)
  assert(${res} ISNULL)

  range_indices("-1" 1)
  ans(res)
  assert(${res} EQUALS 1)

  range_indices("4" :)
  ans(res)
  assert(${res} EQUALS 0 1 2 3)

  range_indices("4" ":)")
  ans(res)
  assert(${res} EQUALS 0 1 2)

  range_indices("4" "(:)")
  ans(res)
  assert(${res} EQUALS 1 2)

  range_indices("-1" 1 9 7 3 5)
  ans(res)
  assert(${res} EQUALS 1 9 7 3 5)


  range_indices("-1" 9:6:-1 10:13 6 4)
  ans(res)
  assert(${res} EQUALS 9 8 7 6 10 11 12 13 6 4)

  range_indices("5" $:0:-1 $ :)
  ans(res)
  assert(${res} EQUALS 4 3 2 1 0 4 0 1 2 3 4)  


  range_indices("5" "(1:6)") # 2 3 4 5
  ans(res)
  assert(${res} EQUALS 2 3 4 5)

  timer_start(timer)
  range_indices(1000 : : : :)
  ans(res)
  timer_print_elapsed(timer)
  list(LENGTH res len)
  assert(${len} EQUALS 4000)


endfunction()