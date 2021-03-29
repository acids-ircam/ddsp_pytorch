function(test)

  
  ## task tests
  task("")
  ans(res)
  assert(NOT res)

  task("message" "a b c" )
  ans(res)
  assert(res)

  task("${res}")
  ans(res2)
  assert(res2)
  assert("${res}" STREQUAL "${res2}")


  ## task anonymous
  task_anonymous("1;3;2" (a b c) return("\${c} \${a} \${b}"))
  ans(res)
  assert(res)



  ## task invoke
  task_anonymous("1;3;2" (a b c) return("\${c} \${a} \${b}"))
  ans(res)
  task_invoke("${res}")
  assertf("{res.return_value}" STREQUAL "2 1 3" )


endfunction()