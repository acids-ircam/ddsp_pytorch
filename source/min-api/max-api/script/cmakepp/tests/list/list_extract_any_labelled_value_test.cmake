function(test)


  set(lstA a b c d e f g)
  list_extract_any_labelled_value(lstA c f)
  ans(res)
  assert(${res} EQUALS d)
  assert(${lstA} EQUALS a b e f g)

endfunction()