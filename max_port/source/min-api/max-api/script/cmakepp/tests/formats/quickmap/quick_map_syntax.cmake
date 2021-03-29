function(test)

  val("asd")



  # empty map
  map()
  end()

  ans(res)
  assert(res)
  is_map(${res} )
  ans(ismap)
  assert(ismap)
  
  # single value
  map()
    key(key1)
    val(val1)
  end()
  
  ans(res)
  assert(DEREF {res.key1} STREQUAL "val1")

  # complex map
  map()
    map(k1)
      key(a)
        val(1)
        val(2)
        val(3)
    end()
    map(k2)
      key(1)
        val(a)
        val(b)
        val(c)
      key(2)
        val(I)
        val(II)
        val(III)
      map(3)
        val(i)
        val(j)
        val(k)
      end()
    end()
  end()
  ans(res)


endfunction()