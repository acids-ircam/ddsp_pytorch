function(test)
  set(thelist 1 0 0)
  list_extract(thelist v1 v2 v3)
  ans(res)
  assert(${res} ISNULL)
  assert(${v1} STREQUAL 1)
  assert(${v2} STREQUAL 0)
  assert(${v3} STREQUAL 0)



  set(thelist 2 3 0 0 1)
  list_extract(thelist v1 v2 v3 v4 v5)
  assert("${v1}" STREQUAL 2)
  assert("${v2}" STREQUAL 3)
  assert("${v3}" STREQUAL 0)
  assert("${v4}" STREQUAL 0)
  assert("${v5}" STREQUAL 1)

  set(val5)
  set(thelist a b c d)
  list_extract(thelist val1 val2 val3 val4 val5)

  assert(${val1} STREQUAL a)
  assert(${val2} STREQUAL b)
  assert(${val3} STREQUAL c)
  assert(${val4} STREQUAL d)
  assert(NOT val5)


  set(thelist 1)
  list_extract(thelist val1 val2)
  assert(${val1} STREQUAL 1)
  assert(NOT val2)

endfunction()