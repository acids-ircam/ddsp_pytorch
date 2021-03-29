function(test)






  uri_params_serialize("{a:{b:{c:1}}}")
  ans(res)
  assert("${res}" STREQUAL "a[b][c]=1")

  uri_params_serialize("{a:1,b:2}")
  ans(res)
  assert(${res} STREQUAL "a=1&b=2")

  


endfunction()