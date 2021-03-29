function(test)



  function(base64_encode str)
    string(LENGTH "${str}" len)
    math(EXPR len "${len} - 1")
    foreach(i RANGE 0 len)
      string_char_at("${str}" ${i})
      ans(c)

      ascii_code("${c}")
      ans(code)



    endforeach()
  endfunction()

  function(base64_decode str)

  endfunction()


endfunction()