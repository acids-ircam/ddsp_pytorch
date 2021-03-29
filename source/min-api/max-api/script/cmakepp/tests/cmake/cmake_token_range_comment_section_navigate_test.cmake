function(test)

  cmake_token_range_comment_section_navigate("

    ## <section name=\"sec1\">


    ## <section name=\"sec2\">\nset(a 1)\n## </section>

    ## </section>

    " "sec1.sec2")


  ans(res)
  assert(res)
  cmake_token_range_serialize("${res}")
  ans(res)
  assert("${res}" STREQUAL "set(a 1)\n")
  _message("${res}")



endfunction()