function(test)


  fwrite_temp("thedata")
  ans(path)

  assert(EXISTS "${path}")
  fread("${path}")
  ans(res)
  assert("${res}" STREQUAL "thedata")


  fwrite_temp("thedata2" ".bat")
  ans(res)
  assert("${res}" MATCHES "\\.bat$")

  

endfunction()