function(test)

  set(lst)
  list_range_partial_write(lst [0] 123)
  assert(${lst} EQUALS 123)

  list_range_partial_write(lst "[1]" 234)
  assert(${lst} EQUALS 123 234)


  list_range_partial_write(lst "[1[" 234)
  assert(${lst} EQUALS 123 234 234)
  endfunction()