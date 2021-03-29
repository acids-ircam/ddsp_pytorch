function(test)

  set(lst a b c d)

  string_encode_list("${lst}")
  ans(res)
  assert(COUNT 1 ${res})


  string_decode_list("${res}")
  ans(res)
  assert(COUNT 4 ${res})


  # empty list encode
  # null lists are not encoded by string_encode_empty anymore
  # string_encode_list("")
  # ans(res)
  # assert(COUNT 1 ${res})

  # string_decode_list("${res}")
  # ans(res)
  # assert(COUNT 0 ${res})



  string_encode_list("a")
  ans(res)
  assert(COUNT 1 ${res}) 

  string_decode_list("${res}")
  ans(res)
  assert("${res}" STREQUAL "a")


  



endfunction()