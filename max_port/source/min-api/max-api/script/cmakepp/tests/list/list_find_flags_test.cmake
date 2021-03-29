function(test)


  set(lst a b c d e f g)
  list_find_flags(lst a d f)
  ans(res)
  assertf("{res.a}")
  assertf("{res.d}")
  assertf("{res.f}")

endfunction()