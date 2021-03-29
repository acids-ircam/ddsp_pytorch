function(test)

  cmake_token_range_variable_navigate("
    ## <section name=\"sec1\">
    ## <section name=\"sec2\">
    set(varA 1 2 3)
    ## </section>
    ## </section>
    " "sec1.sec2.varA" --append 4 5 )
  ans(res)

  assert(${res} EQUALS 1 2 3 4 5)


  

endfunction()