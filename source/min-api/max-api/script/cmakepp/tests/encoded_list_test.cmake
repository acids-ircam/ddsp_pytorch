function(test)



  encoded_list("a;b;c" "c;d;e")
  ans(res)

  is_encoded_list("${res}")
  ans(ok)
  assert(ok)

  encoded_list_get(res 1)
  ans(res)
  assert("${res}" EQUALS c d e)  


  #string_codes()
  #message("${paren_open_code}")
  #message("${paren_close_code}")
  #message("${semicolon_code}")
  
  encoded_list("a;b;c" "c;d;e")
  ans(inp)

  encoded_list_to_cmake_string(${inp})
  ans(res)

  eval("encoded_list(${res})" )
  ans(el)
 
  assert("${el}" EQUALS "${inp}")






endfunction()