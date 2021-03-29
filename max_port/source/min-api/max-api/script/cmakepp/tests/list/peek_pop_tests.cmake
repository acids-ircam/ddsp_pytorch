function(test)

  set(lst1 0 1 2 3)
  set(lst2 3 2 1 0)
  set(lst3 0 0 0 0)
  set(lst4 0)
  set(lst5 )

  # peek front
  list_peek_front( lst1)
  ans(res)
  assert("${res}" STREQUAL "0")

  list_peek_front( lst2)
  ans(res)
  assert("${res}" STREQUAL "3")

  list_peek_front( lst3)
  ans(res)
  assert("${res}" STREQUAL "0")

  list_peek_front( lst4)
  ans(res)
  assert("${res}" STREQUAL "0")

  list_peek_front( lst5)
  ans(res)
  assert(NOT res)  


  # peek back
  list_peek_back( lst1)
  ans(res)
  assert("${res}" STREQUAL "3")

  list_peek_back( lst2)
  ans(res)
  assert("${res}" STREQUAL "0")

  list_peek_back( lst3)
  ans(res)
  assert("${res}" STREQUAL "0")

  list_peek_back( lst4)
  ans(res)
  assert("${res}" STREQUAL "0")

  list_peek_back( lst5)
  ans(res)
  assert(NOT res)  

  # pop front
  list_pop_front( lst1)
  ans(res)
  assert("${res}" STREQUAL 0)
  assert(EQUALS ${lst1} 1 2 3 )
 

  list_pop_front( lst2)
  ans(res)
  assert("${res}" STREQUAL 3)
  assert(EQUALS ${lst2} 2 1 0)

  list_pop_front( lst3)
  ans(res)
  assert("${res}" STREQUAL 0)
  assert(EQUALS ${lst3} 0 0 0  )


  list_pop_front( lst3)
  ans(res)
  assert("${res}" STREQUAL 0)
  assert(EQUALS ${lst3} 0 0  )

  list_pop_front( lst3)
  ans(res)
  assert("${res}" STREQUAL 0)
  assert(EQUALS ${lst3} 0  )

  list_pop_front( lst3)
  ans(res)
  assert("${res}" STREQUAL 0)
  assert(NOT lst3  )




  set(lst1 0 1 2 3)
  set(lst2 3 2 1 0)
  set(lst3 0 0 0 0)

  #pop back
  list_pop_back( lst1)
  ans(res)
  assert("${res}" STREQUAL 3)
  assert(EQUALS ${lst1} 0 1 2 )
 

  list_pop_back( lst2)
  ans(res)
  assert("${res}" STREQUAL 0)
  assert(EQUALS ${lst2} 3 2 1 )

  list_pop_back( lst3)
  ans(res)
  assert("${res}" STREQUAL 0)
  assert(EQUALS ${lst3} 0 0 0   )

  list_pop_back( lst3)
  ans(res)
  assert("${res}" STREQUAL 0)
  assert(EQUALS ${lst3} 0 0   )

  list_pop_back( lst3)
  ans(res)
  assert("${res}" STREQUAL 0)
  assert(EQUALS ${lst3}  0   )


  list_pop_back( lst3)
  ans(res)
  assert("${res}" STREQUAL 0)
  assert(NOT lst3 )


  



endfunction()