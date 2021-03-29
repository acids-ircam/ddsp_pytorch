function(test)

  function_define_new((asd bsd csd) return("\${asd} \${bsd} \${csd}"))
  ans(fu)

  eval_cmake(${fu}(1 "k;k" 3))
  ans(res)
  assert(${res} EQUALS "1 k;k 3")

endfunction()