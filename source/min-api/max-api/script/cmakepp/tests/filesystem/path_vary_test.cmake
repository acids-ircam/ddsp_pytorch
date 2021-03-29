function(test)


  path_vary("asd.txt")
  ans(res)
  
  assert("${res}" STREQUAL "${test_dir}/asd.txt")  

  fwrite("${res}")
  path_vary("asd.txt")
  ans(res)

  assert("${res}" MATCHES "\\/asd\\_.+\\.txt")
  

endfunction()