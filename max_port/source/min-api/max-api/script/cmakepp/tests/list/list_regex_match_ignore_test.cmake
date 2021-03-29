function(test)

  set(lstA a b c d e f g)
  

  list_regex_match_ignore(lstA "[a-e]" "![c]")
  ans(res)
  assert(${res} EQUALS a b d e)

  list_regex_match_ignore(lstA "[a-c]" "[e-g]" "![c-e]" "![g]")
  ans(res)
  assert(${res} EQUALS a b f)

  




endfunction()