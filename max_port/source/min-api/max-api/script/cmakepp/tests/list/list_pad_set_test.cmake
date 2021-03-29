function(test)



  list_pad_set(testList "1;4;5;10" "$" "a")
  assert(${testList} EQUALS $ a $ $ a a $ $ $ $ a)


endfunction()