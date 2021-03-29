function(test)





  set(test_lst 0 1 2 3 4 5 6 7 8 9 a b c d e f)
  list_range_set(test_lst "0:$:2" "+")
  assert(${test_lst} EQUALS + 1 + 3 + 5 + 7 + 9 + b + d + f)

  set(test_lst 0 1 2 3 4 5 6 7 8 9 a b c d e f)
  list_range_set(test_lst "0:$:2" "")
  assert(${test_lst} EQUALS 1 3 5 7 9 b d f)

  set(test_lst 1 2 3 4)


endfunction()