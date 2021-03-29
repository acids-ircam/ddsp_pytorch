function(test)




  range_simplify(9 "[1:$-1]" "[2:5]" "[$:0:-1]")
  ans(res)
  assert(${res} EQUALS "[6:3:-1]")

  range_simplify(9 "1 4 7 2 3 4 1")
  ans(res)
  assert(${res} EQUALS "[1:7:3] [2:4] 1")


endfunction()