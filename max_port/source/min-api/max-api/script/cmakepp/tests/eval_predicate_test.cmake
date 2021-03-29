function(test)


  

  set(lstA a b c)
  set(valA 1)
  set(valB 2)
  set(lstB d e f)
  set(valC)

  eval_predicate(valA STREQUAL 1)
  ans(res)
  assert(res)
  
  eval_predicate(valA STREQUAL 2)
  ans(res)
  assert(NOT res)

  eval_predicate(valB STREQUAL 2)
  ans(res)
  assert(res)

  eval_predicate(valC STREQUAL 3)
  ans(res)
  assert(NOT res)

  eval_predicate(NOT valC STREQUAL 3)
  ans(res)
  assert(res)



  set(pred \${invocation_identifier} STREQUAL label)
  set(invocation_identifier label)
  eval_predicate(${pred})
  ans(res)  
  assert(res)



  set(abc "abc bfc")
  eval_predicate(abc STREQUAL abc\ bfc)
  ans(res)
  assert(res)


  eval_predicate("a;b;c" STREQUAL "a;b;c")
  ans(res)
  assert(res)

  


endfunction()