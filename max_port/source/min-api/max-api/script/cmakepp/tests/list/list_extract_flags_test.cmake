function(test)
  set(lstA a b c d e f)

  list_extract_flags(lstA c e f)
  ans(res)

  assertf("{res.c}")
  assertf("{res.e}")
  assertf("{res.f}")



endfunction()