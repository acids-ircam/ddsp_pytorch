function(test)


  set(listA 0 1 2 3 4 5 6 7 8 9 a b c d e f)


  list_range_get(listA :)
  ans(res)
  assert(${res} EQUALS 0 1 2 3 4 5 6 7 8 9 a b c d e f)

  list_range_get(listA $:0:-1)
  ans(res)
  assert(${res} EQUALS f e d c b a 9 8 7 6 5 4 3 2 1 0)

  list_range_get(listA n/4:n*3/4)
  ans(res)
  assert(${res} EQUALS 4 5 6 7 8 9 a b c)

  list_range_get(listA 11 10 13 15 0 0 13)
  ans(res)
  assert(${res} EQUALS b a d f 0 0 d)

  
endfunction()