function(test)

  ## append to list at the last index
  set(a)
  list_replace_slice(a * * b)
  list_replace_slice(a * * c)
  list_replace_slice(a * * d)
  list_replace_slice(a * * e)
  assert(${a} EQUALS b c d e)


  ## replace the elements starting from first up to last element of list
  set(a k b c d)
  list_replace_slice(a 1 *)
  ans(res)
  assert(${res} EQUALS b c d)
  assert(${a} EQUALS k)
  
  ## append element at the back of list
  set(lstA a b)
  list_replace_slice(lstA -1 -1 c)
  ans(res)
  assert(${res} ISNULL)
  assert(${lstA} EQUALS a b c)


  set(lstA a)
  list_replace_slice(lstA -1 -1 b)
  ans(res)
  assert(${res} ISNULL)
  assert(${lstA} EQUALS a b)

  set(lstA)
  list_replace_slice(lstA -1 -1 a)
  ans(res)
  assert(${res} ISNULL)
  assert(${lstA} EQUALS a)

  set(lstA a b)
  list_replace_slice(lstA * * c)
  ans(res)
  assert(${res} ISNULL)
  assert(${lstA} EQUALS a b c)

  set(lstA a)
  list_replace_slice(lstA * * b)
  ans(res)
  assert(${res} ISNULL)
  assert(${lstA} EQUALS a b)


  set(lstA a b c d e f g)
  list_replace_slice(lstA 2 5)
  ans(res)
  assert(${res} EQUALS c d e)
  assert(${lstA} EQUALS a b f g)

  set(lstA a b c d e f g)
  list_replace_slice(lstA 2 5 1 2 3)
  ans(res)
  assert(${res} EQUALS c d e)
  assert(${lstA} EQUALS a b 1 2 3 f g)

  set(lstA)
  list_replace_slice(lstA -1 -1 a b c)
  ans(res)
  assert(${lstA} EQUALS a b c)
  assert(${res} ISNULL)

  set(lstA a b c)
  list_replace_slice(lstA 1 1 k k)
  ans(res)
  assert(${lstA} EQUALS a k k b c)
  assert(${res} ISNULL)

  set(lstA)
  list_replace_slice(lstA -1 -1)
  ans(res)
  assert(${lstA} ISNULL)
  assert(${res} ISNULL)

  set(lstA a b c)
  list_replace_slice(lstA 0 3 1 2 3 )
  ans(res)
  assert(${res} EQUALS a b c)
  assert(${lstA} EQUALS 1 2 3)


  set(lstA)
  list_replace_slice(lstA 0 0 a b c)
  ans(res)
  assert(${lstA} EQUALS a b c)
  assert(${res} ISNULL)

  set(lst a b c)
  list_replace_slice(lstA 0 * y k u)
  ans(res)
  assert(${res} EQUALS a b c)
  assert(${lstA} EQUALS y k u)




endfunction()