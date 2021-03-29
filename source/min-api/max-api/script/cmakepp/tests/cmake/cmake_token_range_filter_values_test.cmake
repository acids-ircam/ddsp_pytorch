function(test)


  cmake_token_range_filter_values("set(a b c d ef g)" value MATCHES "^[bcdg]$")
  ans(res)
  assert(${res} EQUALS b c d g)

  cmake_token_range_filter_values("set(a b c)" type MATCHES "(command_invocation)|(nesting)")
  ans(res)
  string(REPLACE "(" "1" res "${res}")
  string(REPLACE ")" "2" res "${res}")
  assert(${res} EQUALS set 1 2)

  cmake_token_range_filter_values("set(a b c d)#asdasd\nbsd()#dkdkdkdd\ndadad()" type MATCHES "comment")
  ans(res)
  assert(${res} EQUALS asdasd dkdkdkdd) 
  

  cmake_token_range_filter_values("a b c d" type MATCHES "argument" --take 1 --skip 1 --reverse)
  ans(res)
  assert("${res}" STREQUAL "c")



  

endfunction()