

function(hg_match_refs search)
  hg_get_refs()
  ans(refs)


  list_match(refs "{name:$search}")
  ans(m1)

  list_match(refs "{number:$search}")
  ans(m2)
  list_match(refs "{hash:$search}")
  ans(m3)
  list_match(refs "{type:$search}")
  ans(m4)
  set(res ${m1} ${m2} ${m3} ${m4})
  return_ref(res)
endfunction()
