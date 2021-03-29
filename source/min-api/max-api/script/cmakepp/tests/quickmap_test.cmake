function(test)

  map()
   kv(x x1)
   map(y)
    kv(x x2)
   end()
  end()
  ans(res)
  assertf({res.x} STREQUAL "x1")
  assertf({res.y.x} STREQUAL "x2")




  ref()
  ans(r1)
  map()
  ans(m1)
  key(k1)
  val(v1)
  end()
  ans(m2)
  end()
  ans(r2)

  assert(${r1} STREQUAL ${r2})
  assert(${m1} STREQUAL ${m2})


  ref()
  ans(r1)
  val(123)
  val(234)
  end()
  ans(res)
  

  assert(res)
  assert(${res} EQUALS ${r1})
  address_get(${r1})
  ans(r1)
  assert(${r1} EQUALS 123 234)


  ref()
  ans(r1)
  val(123)
  end()
  ans(res)
  

  assert(res)
  assert(${res} STREQUAL ${r1})

  address_get(${r1})
  ans(r1)
  assert(${r1} STREQUAL 123)



  ref()
  ans(r1)
  end()
  ans(res)

  assert(res)
  assert(r1)
  assert("${res}" STREQUAL "${r1}")




  map()
  ans(m1)
  end()
  ans(res)
  assert(res)
  is_map(${res})
  ans(ismap)
  assert(ismap)
  assert(${m1} STREQUAL ${res})

  map()
  ans(m1)
  key(k1)
  val(v1)
  end()
  ans(res)

  assert(${res} STREQUAL ${m1})
  map_keys(${res})
  ans(keys)
  assert(${keys} EQUALS k1)


  map()
  ans(m1)
  key(k1)
  val(v1)
  key(k2)
  val(v2)
  end()
  ans(res)

  assert(res)
  assert(${res} STREQUAL ${m1})
  map_keys(${res})
  ans(keys)
  assert(${keys} EQUALS k1 k2)



  ref()
  ans(r1)
  map()
  ans(m1)
  end()
  ans(m2)
  end()
  ans(r2)

  assert(${r1} STREQUAL ${r2})
  assert(${m1} STREQUAL ${m2})


  ref()
  ans(r1)
  map()
  ans(m1)
  end()
  ans(m2)
  map()
  ans(m3)
  end()
  ans(m4)
  end()
  ans(r2)

  assert(${r1} STREQUAL ${r2})
  assert(${m1} STREQUAL ${m2})
  assert(${m3} STREQUAL ${m4})
  address_get(${r1})
  ans(vals)
  assert(${vals} CONTAINS ${m1})
  assert(${vals} CONTAINS ${m2})






  map()
    key(x)
    val(123)
  end()
  ans(res)

  assert(res)
  assertf({res.x} STREQUAL 123)

  ref()
  end()
  ans(res)

  assert(res)
  address_get(${res})
  ans(res)
  assert(NOT res)



  ref()
  val(123)
  end()
  ans(res)

  
  assert(res)
  address_get(${res})
  ans(res)
  assert(${res} STREQUAL 123) 



  ref()
  val(123)
  val(456)
  end()

  ans(res)
  assert(res)
  address_get(${res})
  ans(res)
  assert(${res} EQUALS 123 456)

  ref()
    map()
     key(x)
     val(123)
    end()
  end()
  ans(res)

  assert(res)
  address_get(${res})
  ans(res)
  assertf({res.x} STREQUAL 123)




  map()
   key(x)
   val(123)
   key(y)
   val(234)
  end()
  ans(res)

  assert(res)
  assertf({res.x} STREQUAL "123")
  assertf({res.y} STREQUAL "234")



endfunction()